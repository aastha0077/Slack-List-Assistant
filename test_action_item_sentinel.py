from datetime import date, datetime, timedelta, timezone

import pytest

import action_item_sentinel as sentinel
import intent_parser
import project_intelligence


def normalized(item_id, name="Task", *, owner_ids=("U1",), priority="P2",
               due=None, completed=False):
    return project_intelligence.NormalizedTask(
        item_id=item_id, item={"id": item_id}, name=name,
        owner_ids=tuple(owner_ids), priority=priority, due_date=due,
        completed=completed, status="Completed" if completed else "Pending",
        created_date=None)


def test_change_detection_reports_meaningful_transitions():
    today = date(2026, 9, 24)
    before = {
        "priority": sentinel.task_state(normalized("priority", priority="P2")),
        "earlier": sentinel.task_state(normalized("earlier", due=today + timedelta(days=6))),
        "later": sentinel.task_state(normalized("later", due=today + timedelta(days=1))),
        "done": sentinel.task_state(normalized("done")),
        "same": sentinel.task_state(normalized("same")),
    }
    current = [
        normalized("priority", priority="P1"),
        normalized("earlier", due=today + timedelta(days=1)),
        normalized("later", due=today + timedelta(days=6)),
        normalized("done", completed=True),
        normalized("same"),
        normalized("new"),
    ]
    kinds = {(change.task_id, change.change_type)
             for change in sentinel.detect_changes(before, current, 1.0)}
    assert ("priority", "priority_escalated") in kinds
    assert ("earlier", "deadline_moved_earlier") in kinds
    assert ("later", "deadline_moved_later") in kinds
    assert ("done", "completed") in kinds
    assert ("new", "created") in kinds
    assert not any(task_id == "same" for task_id, _ in kinds)


def test_risk_detection_is_deterministic_and_explainable():
    today = date(2026, 9, 24)
    snapshot = [
        normalized("late", "Late", priority="P1", due=today - timedelta(days=1)),
        normalized("soon", "Soon", priority="P1", due=today + timedelta(days=1)),
        normalized("other", "Other", owner_ids=("U1",), priority="P1", due=today + timedelta(days=2)),
        normalized("unassigned", "Unassigned", owner_ids=(), priority="P1", due=today + timedelta(days=2)),
        normalized("normal", "Normal", priority="P3", due=today + timedelta(days=10)),
    ]
    risks = sentinel.detect_risks(snapshot, today)
    by_id = {risk.task_id: risk for risk in risks}
    assert by_id["late"].risk_type == "overdue"
    assert "Still pending" in by_id["soon"].reasons
    assert by_id["unassigned"].risk_type == "unassigned_deadline_risk"
    assert by_id["owner:U1"].risk_type == "combined_workload_risk"
    assert "normal" not in by_id


def test_sentinel_adds_non_executable_emerging_risk_without_duplicate_current_risk():
    today = date(2026, 9, 24)
    snapshot = [
        normalized("cluster-a", "Cluster A", priority="P1", due=today + timedelta(days=4)),
        normalized("cluster-b", "Cluster B", priority="P1", due=today + timedelta(days=4)),
        normalized("current", "Current", priority="P1", due=today + timedelta(days=1)),
    ]
    risks = sentinel.detect_risks(snapshot, today)
    by_id = {risk.task_id: risk for risk in risks}
    assert by_id["cluster-a"].risk_type == "emerging_predictive_risk"
    assert "2 tasks share this deadline" in by_id["cluster-a"].reasons
    assert by_id["current"].risk_type == "deadline_risk"
    assert sum("current" in risk.task_ids for risk in risks) == 1


def test_persistent_dedup_new_state_and_resolution(tmp_path):
    store = sentinel.SentinelStore(str(tmp_path / "sentinel.sqlite3"))
    engine = sentinel.ActionItemSentinel(store)
    now = datetime(2026, 9, 24, 9, tzinfo=timezone.utc)
    risky = [normalized("one", priority="P1", due=now.date())]
    first = engine.evaluate("L1", risky, now)
    second = engine.evaluate("L1", risky, now + timedelta(minutes=5))
    assert len(first.emitted) == 1
    assert len(second.emitted) == 0
    assert second.suppressed == 1

    changed = [normalized("one", priority="P1", due=now.date() - timedelta(days=1))]
    third = engine.evaluate("L1", changed, now + timedelta(hours=1))
    assert any(alert.event_type == "risk:overdue" for alert in third.emitted)
    engine.evaluate("L1", [normalized("one", priority="P1", due=now.date(), completed=True)],
                    now + timedelta(hours=2))
    assert not [alert for alert in store.alerts("L1", status="active")
                if alert.event_type.startswith("risk:")]


def test_sentinel_commands_are_local_and_structured(monkeypatch):
    monkeypatch.setattr(intent_parser, "OLLAMA_API_KEY", "")
    assert intent_parser.parse_intent("what requires my attention?")["assignee_self"] is True
    assert intent_parser.parse_intent("detect emerging risks")["intent"] == "sentinel"
    assert intent_parser.parse_intent("show sentinel alerts")["sentinel_mode"] == "alerts"
    action = intent_parser.parse_intent("send sentinel alert 2")
    assert (action["sentinel_action"], action["selection_index"]) == ("approve", 2)
    changed = intent_parser.parse_intent("what changed this week?")
    assert changed["intent"] == "history"
    assert changed["history_period"] == "this_week"
    assert changed.get("task_name") is None


@pytest.mark.parametrize("text,action", [
    ("send sentinel alert 1", "approve"),
    ("send alert 1", "approve"),
    ("approve sentinel alert 1", "approve"),
    ("send reminder for alert 1", "approve"),
    ("dismiss sentinel alert 1", "dismiss"),
    ("dismiss alert 1", "dismiss"),
])
def test_sentinel_action_variants_have_highest_precedence(text, action):
    parsed = intent_parser.parse_intent(text)
    assert parsed["intent"] == "sentinel"
    assert parsed["sentinel_mode"] == "action"
    assert parsed["sentinel_action"] == action
    assert parsed["selection_index"] == 1


def test_sentinel_action_requires_a_valid_number_and_does_not_capture_unrelated_text():
    missing = intent_parser.parse_intent("send sentinel alert")
    assert missing["intent"] == "clarify"
    assert "number" in missing["clarification"]
    zero = intent_parser.parse_intent("send alert 0")
    assert zero["intent"] == "sentinel"
    assert zero["selection_index"] == 0
    assert intent_parser.parse_intent("send the client report")["intent"] != "sentinel"
