from datetime import date, datetime, time, timedelta, timezone

import pytest

import intent_parser
import operations_intelligence
import project_intelligence
import team_calendar
import visual_analytics


TODAY = date(2026, 10, 4)


def task(item_id, name, *, owners=(), priority="P3", due=None, completed=False):
    return project_intelligence.NormalizedTask(
        item_id=item_id, item={"id": item_id}, name=name,
        owner_ids=tuple(owners), priority=priority, due_date=due,
        completed=completed, status="Completed" if completed else "Pending",
        created_date=TODAY - timedelta(days=10))


@pytest.mark.parametrize("phrase,mode", [
    ("give me my daily briefing", "briefing"),
    ("give me an executive summary", "executive"),
    ("who is overloaded?", "workload"),
    ("who has the most overdue work?", "workload"),
    ("what are our biggest risks?", "risk"),
    ("show task health", "health"),
    ("show deadline heatmap", "heatmap"),
    ("where are the bottlenecks?", "bottlenecks"),
    ("which tasks are unassigned?", "unassigned"),
    ("which dates have deadline collisions?", "collisions"),
    ("when can I meet with Praveen?", "meeting"),
])
def test_operations_queries_route_deterministically(phrase, mode, monkeypatch):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("LLM called"))
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == "operations_intelligence"
    assert parsed["operations_mode"] == mode


@pytest.mark.parametrize("phrase,theme", [
    ("show testing work", "testing"),
    ("show deployment work", "deployment"),
    ("show documentation work", "documentation"),
])
def test_workstream_queries_reuse_deterministic_task_search(phrase, theme, monkeypatch):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("LLM called"))
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == "list"
    assert parsed["query"] == theme
    assert parsed["search"] is True


def test_risk_health_and_reasons_are_deterministic():
    values = [
        task("1", "Critical work", owners=("UA",), priority="P1", due=TODAY - timedelta(days=2)),
        task("2", "Healthy work", owners=("UP",), priority="P3", due=TODAY + timedelta(days=20)),
    ]
    risks = operations_intelligence.assess_risks(values, TODAY)
    assert risks[0].level == "Critical"
    assert operations_intelligence.task_health(risks[0]) == "Critical"
    assert "Overdue by 2 days" in risks[0].reasons
    assert "P1 priority" in risks[0].reasons
    assert operations_intelligence.task_health(risks[-1]) == "Healthy"


def test_workload_bottlenecks_unassigned_and_collisions():
    values = [
        task("1", "One", owners=("UA",), priority="P1", due=TODAY + timedelta(days=1)),
        task("2", "Two", owners=("UA",), priority="P1", due=TODAY + timedelta(days=1)),
        task("3", "Three", owners=("UA",), priority="P2", due=TODAY + timedelta(days=1)),
        task("4", "No owner", priority="P1", due=TODAY + timedelta(days=2)),
        task("5", "Light", owners=("UP",), priority="P3", due=TODAY + timedelta(days=20)),
    ]
    summary = __import__("predictive_intelligence").build_predictive_summary(values, TODAY)
    labels = operations_intelligence.workload_labels(summary.workload)
    assert labels["UA"] == "High"
    found = operations_intelligence.bottlenecks(
        values, TODAY, lambda value: {"UA": "AasthaA", "UP": "Praveen"}[value])
    assert any(item.title == "AasthaA" for item in found)
    assert any(item.title == "Unassigned high-priority work" for item in found)
    assert any(item.title.startswith("Deadline cluster") for item in found)


def test_heatmap_calculation_and_visual_artifact():
    values = [
        task("1", "One", priority="P1", due=TODAY + timedelta(days=1)),
        task("2", "Two", priority="P2", due=TODAY + timedelta(days=1)),
    ]
    points = operations_intelligence.heatmap(values, TODAY)
    assert points == ((TODAY + timedelta(days=1), 2, 1),)
    png = visual_analytics.render_deadline_heatmap_png(points, today=TODAY)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 15_000


def test_meeting_windows_require_complete_configuration():
    clocks = [
        team_calendar.MemberClock("UA", "AasthaA", "Asia/Kathmandu", "Nepal",
                                  datetime(2026, 10, 5, 10, tzinfo=timezone(timedelta(hours=5, minutes=45))),
                                  "UTC+5:45", time(9), time(18), "Working hours"),
        team_calendar.MemberClock("UP", "Praveen", "Asia/Kolkata", "India",
                                  datetime(2026, 10, 5, 9, 45, tzinfo=timezone(timedelta(hours=5, minutes=30))),
                                  "UTC+5:30", time(9), time(18), "Working hours"),
    ]
    windows = operations_intelligence.meeting_windows(clocks, clocks[0])
    assert len(windows) == 1
    assert windows[0][1] - windows[0][0] == timedelta(hours=1)
    incomplete = [clocks[0], team_calendar.MemberClock(
        "UM", "Morgan", None, None, None, None, None, None, "Working hours not configured")]
    assert operations_intelligence.meeting_windows(incomplete, clocks[0]) == ()
