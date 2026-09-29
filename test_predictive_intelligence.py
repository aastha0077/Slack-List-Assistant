from datetime import date, timedelta

import predictive_intelligence as intelligence
import project_intelligence


TODAY = date(2026, 9, 29)


def task(item_id, *, name=None, owner=("U1",), priority="P3", due=None, completed=False):
    return project_intelligence.NormalizedTask(
        item_id=item_id, item={"id": item_id}, name=name or item_id,
        owner_ids=tuple(owner), priority=priority, due_date=due,
        completed=completed, status="Completed" if completed else "Pending",
        created_date=None)


def test_predictive_summary_calculates_deadline_and_priority_pressure():
    snapshot = [
        task("late", priority="P1", due=TODAY - timedelta(days=2)),
        task("today", priority="P2", due=TODAY),
        task("tomorrow", priority="P1", due=TODAY + timedelta(days=1)),
        task("week", priority="P3", due=TODAY + timedelta(days=6)),
        task("done", priority="P1", due=TODAY, completed=True),
    ]
    summary = intelligence.build_predictive_summary(snapshot, TODAY)
    assert (summary.pending, summary.completed, summary.overdue) == (4, 1, 1)
    assert (summary.due_today, summary.due_within_24h,
            summary.due_within_48h, summary.due_within_7d) == (1, 2, 2, 3)
    assert summary.priority_counts == {"P1": 2, "P2": 1, "P3": 1}


def test_workload_pressure_counts_each_owner_and_unassigned_work():
    snapshot = [
        task("one", owner=("U1",), priority="P1", due=TODAY + timedelta(days=1)),
        task("two", owner=("U1",), priority="P2", due=TODAY + timedelta(days=5)),
        task("none", owner=(), priority="P1", due=TODAY + timedelta(days=2)),
    ]
    rows = {row.owner_id: row for row in intelligence.calculate_workload_pressure(snapshot, TODAY)}
    assert (rows["U1"].pending, rows["U1"].p1, rows["U1"].due_within_48h) == (2, 1, 1)
    assert (rows[None].pending, rows[None].p1) == (1, 1)


def test_deadline_concentration_groups_only_near_term_pending_tasks():
    snapshot = [
        task("one", priority="P1", due=TODAY + timedelta(days=3)),
        task("two", owner=("U2",), priority="P2", due=TODAY + timedelta(days=3)),
        task("far", due=TODAY + timedelta(days=10)),
        task("done", due=TODAY + timedelta(days=3), completed=True),
    ]
    clusters = intelligence.calculate_deadline_concentration(snapshot, TODAY)
    assert len(clusters) == 1
    assert clusters[0].task_ids == ("one", "two")
    assert clusters[0].priority_counts == {"P1": 1, "P2": 1}


def test_emerging_risk_uses_evidence_and_excludes_overdue_completed_and_normal_future():
    snapshot = [
        task("late", priority="P1", due=TODAY - timedelta(days=1)),
        task("soon", priority="P1", due=TODAY + timedelta(days=1)),
        task("normal", priority="P3", due=TODAY + timedelta(days=6)),
        task("done", priority="P1", due=TODAY + timedelta(days=1), completed=True),
    ]
    risks = intelligence.detect_emerging_risks(snapshot, TODAY)
    assert [risk.task_id for risk in risks] == ["soon"]
    assert risks[0].level == "high"
    assert risks[0].evidence[:2] == ("1 day until deadline", "P1 priority")


def test_cluster_and_owner_pressure_create_explainable_emerging_evidence():
    due = TODAY + timedelta(days=4)
    snapshot = [
        task("one", priority="P1", due=due),
        task("two", priority="P1", due=due),
    ]
    risks = intelligence.detect_emerging_risks(snapshot, TODAY)
    assert {risk.task_id for risk in risks} == {"one", "two"}
    assert all("Owner has 2 pending P1 tasks" in risk.evidence for risk in risks)
    assert all("2 tasks share this deadline" in risk.evidence for risk in risks)


def test_workload_forecast_is_a_projection_of_known_tasks_only():
    summary = intelligence.build_predictive_summary([
        task("one", owner=("U1",), due=TODAY + timedelta(days=1)),
        task("two", owner=(), due=TODAY + timedelta(days=5)),
    ], TODAY)
    lines = intelligence.workload_forecast(summary, lambda value: {"U1": "Praveen"}[value])
    assert "Praveen · 1 pending · 1 due <48h · 1 due in 7 days" in lines
    assert "Unassigned · 1 pending · 0 due <48h · 1 due in 7 days" in lines


def test_analysis_never_mutates_snapshot():
    snapshot = [task("one", priority="P1", due=TODAY + timedelta(days=1))]
    before = tuple(snapshot)
    intelligence.build_predictive_summary(snapshot, TODAY)
    assert tuple(snapshot) == before
