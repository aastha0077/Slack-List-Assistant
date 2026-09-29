from datetime import date, timedelta

import command_center
import project_intelligence


TODAY = date(2026, 9, 25)


def task(item_id, name, *, owner=(), priority="P3", due=None, completed=False):
    return project_intelligence.NormalizedTask(
        item_id=item_id, item={"id": item_id}, name=name, owner_ids=tuple(owner),
        priority=priority, due_date=due, completed=completed,
        status="Completed" if completed else "Pending", created_date=None)


def test_command_center_report_uses_real_snapshot_counts_and_risks():
    snapshot = [
        task("late", "Late", owner=("UP",), priority="P1", due=TODAY - timedelta(days=1)),
        task("today", "Today", owner=("UA",), priority="P2", due=TODAY),
        task("future", "Future", owner=("UP",), priority="P3", due=TODAY + timedelta(days=4)),
        task("done", "Done", owner=("UA",), priority="P1", completed=True),
    ]
    report = command_center.build_report(
        snapshot, today=TODAY, name_for_user={"UP": "Praveen", "UA": "AasthaA"}.get)
    assert (report.pending, report.completed, report.overdue, report.due_this_week) == (3, 1, 1, 2)
    assert report.priorities == {"P1": 1, "P2": 1, "P3": 1}
    assert [value.name for value in report.critical] == ["Late", "Today"]
    assert report.reminder_count >= 1
    assert report.workload.rows["UP"]["pending"] == 2
    assert report.predictive.pending == 3
    assert report.predictive.due_within_7d == 2


def test_command_center_empty_snapshot_is_truthful():
    report = command_center.build_report([], today=TODAY, name_for_user=lambda value: value)
    assert report.pending == report.completed == report.overdue == 0
    assert not report.critical
    assert not report.risks
    assert report.next_step == "No pending action is required."


def test_owner_risks_only_returns_selected_owner_facts():
    snapshot = [
        task("p", "Praveen risk", owner=("UP",), priority="P1", due=TODAY),
        task("a", "Aastha risk", owner=("UA",), priority="P1", due=TODAY),
    ]
    owned, risks = command_center.owner_risks(snapshot, "UP", today=TODAY)
    assert [value.item_id for value in owned] == ["p"]
    assert risks
    assert all("a" not in risk.task_ids for risk in risks)


def test_command_center_workload_failure_degrades_without_losing_counts(monkeypatch):
    monkeypatch.setattr(
        project_intelligence, "calculate_workload",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated")))
    report = command_center.build_report(
        [task("one", "One", priority="P2")], today=TODAY,
        name_for_user=lambda value: value)
    assert report.pending == 1
    assert report.workload.rows == {}
