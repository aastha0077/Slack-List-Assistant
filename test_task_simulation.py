from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import decision_ledger
import main
import project_intelligence
import pytest
import task_simulation


TODAY = date(2026, 9, 27)


def task(item_id, name, *, owners=(), priority="P3", due=None, completed=False):
    return project_intelligence.NormalizedTask(
        item_id=item_id, item={"id": item_id}, name=name,
        owner_ids=tuple(owners), priority=priority, due_date=due,
        completed=completed, status="Completed" if completed else "Pending",
        created_date=TODAY - timedelta(days=5),
    )


def base_tasks():
    return [
        task("T1", "Unassigned release", priority="P1", due=TODAY),
        task("T2", "Praveen work", owners=("UP",), priority="P1", due=TODAY + timedelta(days=1)),
        task("T3", "Aastha work", owners=("UA",), due=TODAY + timedelta(days=5)),
    ]


def simulate(operation, task_ids=("T1",), parameters=None):
    return task_simulation.simulate(
        requester_id="U", goal="Test scenario", operation=operation,
        tasks=base_tasks(), task_ids=task_ids, parameters=parameters or {},
        today=TODAY, now=1000,
    )


def test_deterministic_scenario_parsing():
    parsed = task_simulation.parse_request(
        "what happens if I assign the unassigned P1 task to Praveen?", TODAY)
    assert parsed["intent"] == "simulation"
    assert parsed["scenario"]["operation"] == "assign_task"
    assert parsed["scenario"]["target_unassigned"] is True
    assert parsed["scenario"]["assignee_names"] == ("Praveen",)


@pytest.mark.parametrize("phrase", [
    "what happens if I assign the unassigned P1 task to Praveen?",
    "what happens if I assign the unassigned P1 to Praveen?",
    "simulate assigning the unassigned P1 task to Praveen",
    "simulate assigning the unassigned P1 to Praveen",
    "what if I give the unassigned P1 task to Praveen?",
    "what if we assign the unassigned P1 task to Praveen?",
])
def test_assignment_scenario_variants_extract_semantic_selector(phrase):
    scenario = task_simulation.parse_request(phrase, TODAY)["scenario"]
    assert scenario["operation"] == "assign_task"
    assert scenario["target_assignee"] == "Praveen"
    assert scenario["assignee_names"] == ("Praveen",)
    assert scenario["target_unassigned"] is True
    assert scenario["target_priority"] == "P1"
    assert scenario["task_reference"] is None


def test_comparison_assignment_extracts_selector_and_both_assignees():
    parsed = task_simulation.parse_request(
        "compare assigning the unassigned P1 task to Praveen vs AasthaA", TODAY)
    scenario = parsed["scenario"]
    assert parsed["simulation_mode"] == "compare"
    assert scenario["operation"] == "assign_task"
    assert scenario["target_unassigned"] is True
    assert scenario["target_priority"] == "P1"
    assert scenario["target_assignee"] == "Praveen"
    assert scenario["assignee_names"] == ("Praveen", "AasthaA")


@pytest.mark.parametrize("selector,expected", [
    ("overdue P1 task", {"target_overdue": True, "target_priority": "P1"}),
    ("overdue tasks", {"target_overdue": True, "selector_plural": True}),
    ("my P1 task", {"target_owner_self": True, "target_priority": "P1"}),
    ("Praveen's P1 task", {"target_owner_name": "Praveen", "target_priority": "P1"}),
    ("task due today", {"target_due": "today"}),
    ("task due tomorrow", {"target_due": "tomorrow"}),
    ("tasks due within 48 hours", {"target_due": "within_48h", "selector_plural": True}),
])
def test_semantic_selectors_are_structured(selector, expected):
    scenario = task_simulation.parse_request(
        f"simulate completing {selector}", TODAY)["scenario"]
    assert scenario["task_reference"] is None
    for key, value in expected.items():
        assert scenario[key] == value


def test_semantic_target_resolution_uses_authorized_snapshot(monkeypatch):
    tasks = [
        task("U1", "Unassigned urgent", priority="P1", due=TODAY - timedelta(days=1)),
        task("P1", "Praveen today", owners=("UP",), priority="P1", due=TODAY),
        task("P2", "Praveen tomorrow", owners=("UP",), priority="P2", due=TODAY + timedelta(days=1)),
        task("ME", "My soon task", owners=("UM",), priority="P1", due=TODAY + timedelta(days=2)),
    ]
    monkeypatch.setattr(main, "current_date", lambda: TODAY)
    monkeypatch.setattr(main.slack_tools, "find_user_id",
                        lambda name: {"praveen": "UP"}.get(name.casefold()))
    ctx = SimpleNamespace(user_id="UM")

    def resolve(**selector):
        request = {"operation": "complete_task", "task_reference": None, **selector}
        selected, _ = main._simulation_targets(request, tasks, {}, ctx, None)
        return [value.item_id for value in selected]

    assert resolve(target_unassigned=True, target_priority="P1") == ["U1"]
    assert resolve(target_overdue=True, target_priority="P1") == ["U1"]
    assert resolve(target_owner_self=True, target_priority="P1") == ["ME"]
    assert resolve(target_owner_name="Praveen", target_priority="P1") == ["P1"]
    assert resolve(target_due="today") == ["P1"]
    assert resolve(target_due="tomorrow") == ["P2"]
    assert resolve(target_due="within_48h", selector_plural=True) == ["P1", "P2", "ME"]


def test_singular_semantic_selector_never_guesses(monkeypatch):
    tasks = [
        task("U1", "First unassigned", priority="P1"),
        task("U2", "Second unassigned", priority="P1"),
    ]
    monkeypatch.setattr(main, "current_date", lambda: TODAY)
    request = {"operation": "assign_task", "task_reference": None,
               "target_unassigned": True, "target_priority": "P1",
               "assignee_names": ("Praveen",)}
    with pytest.raises(ValueError, match="2 matching tasks.*Please name exactly one"):
        main._simulation_targets(
            request, tasks, {}, SimpleNamespace(user_id="UM"), None)


def test_assignment_simulation_clones_without_mutating_source():
    tasks = base_tasks()
    projected = task_simulation.project(tasks, "assign_task", ("T1",), {"assignee_ids": ["UP"]})
    assert tasks[0].owner_ids == ()
    assert projected[0].owner_ids == ("UP",)
    assert projected[0] is not tasks[0]


def test_assignment_simulation_reports_workload_and_tradeoff():
    result = simulate("assign_task", parameters={"assignee_ids": ["UP"]})
    assert result.impact["unassigned_delta"] == -1
    assert result.impact["owner_delta"]["pending"] == 1
    assert "The selected owner's pending workload increases" in result.tradeoffs


def test_due_priority_completion_and_do_nothing_projections():
    due = simulate("change_due_date", parameters={"due_date": (TODAY + timedelta(days=4)).isoformat()})
    priority = simulate("change_priority", task_ids=("T3",), parameters={"priority": "P1"})
    complete = simulate("complete_task", parameters={"completed": True})
    unchanged = simulate("leave_unchanged", task_ids=())
    assert due.simulated_metrics["due_today"] == 0
    assert priority.simulated_metrics["priorities"]["P1"] == 3
    assert complete.impact["completed_delta"] == 1
    assert unchanged.baseline_metrics == unchanged.simulated_metrics


def test_scenario_fingerprint_is_idempotent_for_same_state():
    first = simulate("assign_task", parameters={"assignee_ids": ["UP"]})
    second = simulate("assign_task", parameters={"assignee_ids": ["UP"]})
    assert first.fingerprint == second.fingerprint
    assert first.scenario_id == second.scenario_id


def test_stale_scenario_detection():
    result = simulate("assign_task", parameters={"assignee_ids": ["UP"]})
    assert not task_simulation.is_stale(result, base_tasks())
    changed = [replace(value, priority="P2") if value.item_id == "T1" else value
               for value in base_tasks()]
    assert task_simulation.is_stale(result, changed)


def test_decision_ledger_is_persistent_and_idempotent(tmp_path):
    path = tmp_path / "state.sqlite3"
    result = simulate("assign_task", parameters={"assignee_ids": ["UP"]})
    first = decision_ledger.DecisionLedger(str(path)).create(result, "L")
    second = decision_ledger.DecisionLedger(str(path)).create(result, "L")
    assert first.decision_id == second.decision_id
    rows = decision_ledger.DecisionLedger(str(path)).recent("U", "L")
    assert len(rows) == 1 and rows[0]["scenario"].impact["unassigned_delta"] == -1


def test_ledger_scopes_decisions_to_requester_and_list(tmp_path):
    ledger = decision_ledger.DecisionLedger(str(tmp_path / "state.sqlite3"))
    stored = ledger.create(simulate("assign_task", parameters={"assignee_ids": ["UP"]}), "L")
    assert ledger.get(stored.decision_id, "U", "L")
    assert ledger.get(stored.decision_id, "OTHER", "L") is None
    assert ledger.get(stored.decision_id, "U", "OTHER") is None


def test_ledger_records_expected_vs_actual(tmp_path):
    ledger = decision_ledger.DecisionLedger(str(tmp_path / "state.sqlite3"))
    stored = ledger.create(simulate("assign_task", parameters={"assignee_ids": ["UP"]}), "L")
    actual = stored.simulated_metrics
    ledger.record_outcome(stored.decision_id, "U", "L", actual, True)
    record = ledger.get(stored.decision_id, "U", "L")
    assert record["verification_status"] == "verified"
    assert record["actual"] == actual
