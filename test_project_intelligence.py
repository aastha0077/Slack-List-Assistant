from datetime import date, timedelta
from types import SimpleNamespace

import audit_log
import project_intelligence as intelligence
import progress_engine
import visualization
import workflow_safety


SCHEMA = {"schema": [
    {"id": "name", "key": "name", "name": "Name", "type": "text"},
    {"id": "done", "key": "todo_completed", "name": "Completed", "type": "checkbox"},
    {"id": "owner", "key": "todo_assignee", "name": "Assignee", "type": "user"},
    {"id": "due", "key": "todo_due_date", "name": "Due Date", "type": "date"},
    {"id": "priority", "key": "priority", "name": "Priority", "type": "select",
     "options": {"choices": [{"id": f"priority_{number}", "label": f"P{number}"}
                              for number in range(1, 5)]}},
]}


def task(item_id, name, *, completed=False, assignee=None, priority=None, due=None):
    fields = [
        {"column_id": "name", "text": name},
        {"column_id": "done", "checkbox": completed},
    ]
    if assignee:
        fields.append({"column_id": "owner", "user": [assignee]})
    if priority:
        fields.append({"column_id": "priority", "select": ["priority_" + priority[-1]]})
    if due:
        fields.append({"column_id": "due", "date": [due.isoformat() if isinstance(due, date) else due]})
    return {"id": item_id, "fields": fields}


def test_health_classification_is_factual_and_explainable():
    today = date(2026, 9, 21)
    items = [
        task("late", "Late", due=today - timedelta(days=2), priority="P1"),
        task("soon", "Soon", due=today + timedelta(days=1), priority="P1"),
        task("none", "No deadline", priority="P2"),
        task("later", "Later", due=today + timedelta(days=10), priority="P3"),
    ]
    health = {record.item_id: record for record in intelligence.calculate_health(items, SCHEMA, today)}
    assert health["late"].level == "Overdue"
    assert "Overdue by 2 days" in health["late"].reasons
    assert health["soon"].level == "Needs Attention"
    assert health["soon"].reasons == ("Due tomorrow", "P1 priority", "Still pending")
    assert health["none"].level == "No Deadline"
    assert health["later"].level == "On Track"


def test_planning_returns_exact_ids_without_mutating_tasks():
    today = date(2026, 9, 21)
    items = [
        task("normal", "Normal", due=today + timedelta(days=4), priority="P3"),
        task("urgent", "Urgent", due=today + timedelta(days=1), priority="P1"),
        task("done", "Done", completed=True, due=today, priority="P1"),
    ]
    before = repr(items)
    plan = intelligence.build_plan(items, SCHEMA, today, today + timedelta(days=4), today)
    assert [entry.item_id for entry in plan] == ["urgent", "normal"]
    assert [entry.scheduled_date for entry in plan] == [today, today + timedelta(days=1)]
    assert repr(items) == before


def test_workload_detects_imbalance_and_proposes_exact_existing_item():
    today = date(2026, 9, 21)
    items = [task(f"a{i}", f"A{i}", assignee="UA", priority="P1" if i < 2 else "P3") for i in range(6)]
    items += [task("b1", "B1", assignee="UB", priority="P3")]
    report = intelligence.calculate_workload(items, SCHEMA, lambda uid: uid, today, ["UA", "UB"])
    assert report.rows["UA"]["pending"] == 6
    assert report.rows["UA"]["p1"] == 2
    assert report.overloaded == ["UA"]
    assert report.suggestions[0]["item_id"] in {f"a{i}" for i in range(6)}
    assert report.suggestions[0]["to_user_id"] == "UB"


def test_standup_never_calls_current_completion_state_completed_today():
    today = date(2026, 9, 21)
    completed = task("done", "Done", completed=True)
    report = intelligence.build_standup([completed], SCHEMA, today)
    assert report.completed == [completed]
    assert report.completion_is_daily is False
    assert "not as completed today" in report.limitations[0]
    completed["completed_at"] = today.isoformat() + "T09:00:00Z"
    report = intelligence.build_standup([completed], SCHEMA, today)
    assert report.completion_is_daily is True
    assert report.completed == [completed]


def test_dependency_support_uses_only_explicit_schema_fields():
    schema = {"schema": [*SCHEMA["schema"],
                         {"id": "deps", "key": "depends_on", "name": "Depends On", "type": "text"}]}
    item = task("B", "Task B")
    item["fields"].append({"column_id": "deps", "text": "Task A"})
    assert intelligence.dependency_values(item, schema) == ("Task A",)
    assert intelligence.dependency_values(item, SCHEMA) == ()


def test_duplicate_detection_is_strong_but_does_not_block_weak_similarity():
    items = [task("one", "Prepare client report"), task("two", "Prepare API tests")]
    assert [item["id"] for item in workflow_safety.likely_duplicates(
        "prepare client reports", items, SCHEMA)] == ["one"]
    assert workflow_safety.likely_duplicates("Client call", items, SCHEMA) == []


def test_snapshot_fingerprint_changes_when_any_target_field_changes():
    item = task("one", "Alpha", priority="P2")
    before = workflow_safety.snapshot_fingerprint([item], SCHEMA, ["one"])
    item["fields"].append({"column_id": "due", "date": ["2026-09-25"]})
    after = workflow_safety.snapshot_fingerprint([item], SCHEMA, ["one"])
    assert before != after


def test_audit_history_records_actor_context_and_before_after(tmp_path):
    db_path = str(tmp_path / "audit.sqlite3")
    before = task("one", "Alpha", priority="P2")
    after = task("one", "Alpha", priority="P1")
    ctx = SimpleNamespace(team_id="W", channel_id="C", thread_ts="T", user_id="UA",
                          role="admin", list_id="L")
    audit_log.record(db_path, ctx, "one", "update", [{"field": "priority", "value": "P1"}],
                     SCHEMA, before, after)
    history = audit_log.history(db_path, "L", ["one"])
    assert len(history) == 1
    assert history[0]["actor_id"] == "UA"
    assert history[0]["before"]["priority"] == "P2"
    assert history[0]["after"]["priority"] == "P1"


def test_status_comparison_data_is_structured_before_visualization():
    tasks = [task("p1", "Pending one"), task("p2", "Pending two"),
             task("c1", "Completed", completed=True)]
    report = progress_engine.calculate_progress(
        tasks, SCHEMA, today=date(2026, 9, 21), metrics=["status_distribution"],
        requested_statuses=["open", "completed"])
    assert report.status_distribution == {"Pending": 2, "Completed": 1}
    rendered = visualization.render_progress(report, lambda records, title: title)
    assert "Pending:" in rendered and " 2" in rendered
    assert "Completed:" in rendered and " 1" in rendered
