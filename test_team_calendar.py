from datetime import date, datetime, time, timedelta, timezone

import pytest

import intent_parser
import project_intelligence
import team_calendar
import visual_analytics


TODAY = date(2026, 10, 4)


def task(item_id, name, *, owners=(), priority="P3", due=None, completed=False):
    return project_intelligence.NormalizedTask(
        item_id=item_id, item={"id": item_id}, name=name,
        owner_ids=tuple(owners), priority=priority, due_date=due,
        completed=completed, status="Completed" if completed else "Pending",
        created_date=TODAY - timedelta(days=3))


@pytest.mark.parametrize("phrase,mode", [
    ("show team calendar", "team"),
    ("show calendar", "week"),
    ("show overdue calendar", "overdue"),
    ("show team time zones", "clock"),
    ("who is working right now", "availability"),
    ("when can I meet with the team", "coordination"),
])
def test_calendar_requests_route_deterministically_without_llm(phrase, mode, monkeypatch):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("LLM called"))
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == "calendar"
    assert parsed["calendar_mode"] == mode


def test_existing_upcoming_and_deadline_pressure_routes_are_preserved():
    assert intent_parser.parse_intent("show upcoming deadlines")["intent"] == "visual_analytics"
    assert intent_parser.parse_intent("show deadline pressure")["intent"] == "intelligence_summary"


def test_calendar_filters_and_renders_deadline_intelligence():
    tasks = [
        task("1", "Late P1", owners=("UA",), priority="P1", due=TODAY - timedelta(days=1)),
        task("2", "Today P2", owners=("UP",), priority="P2", due=TODAY),
        task("3", "Soon P1", owners=("UA",), priority="P1", due=TODAY + timedelta(days=2)),
        task("4", "Done", completed=True, due=TODAY),
    ]
    assert [value.name for value in team_calendar.filter_tasks(tasks, "overdue", TODAY)] == ["Late P1"]
    rendered = team_calendar.render_calendar(
        tasks, tasks, "team", TODAY, lambda value: {"UA": "AasthaA", "UP": "Praveen"}[value])
    assert rendered.startswith("*TEAM CALENDAR — TEAM*")
    assert "Late P1" in rendered and "Today P2" in rendered and "Soon P1" in rendered
    assert "*DEADLINE INTELLIGENCE*" in rendered
    assert "No task changes were made" in rendered
    assert "Done" not in rendered
    assert not any(symbol in rendered for symbol in ("📅", "⚠️", "🔴", "🟠", "🟢"))


def test_team_clock_uses_only_configured_or_slack_timezone_data(monkeypatch):
    monkeypatch.setenv("TEAM_TIMEZONES_JSON", '{"UA":"Asia/Kathmandu","UP":"America/New_York"}')
    monkeypatch.setenv("TEAM_LOCATIONS_JSON", '{"UA":"Nepal"}')
    monkeypatch.setenv(
        "TEAM_WORKING_HOURS_JSON",
        '{"UA":{"start":"09:00","end":"18:00"},"UP":"09:00-18:00"}')
    clocks = team_calendar.member_clocks([
        {"id": "UA", "name": "AasthaA"},
        {"id": "UP", "name": "Praveen"},
        {"id": "UM", "name": "Morgan"},
    ], datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc))
    by_id = {clock.user_id: clock for clock in clocks}
    assert by_id["UA"].utc_offset == "UTC+5:45"
    assert by_id["UA"].location == "Nepal"
    assert by_id["UP"].utc_offset == "UTC−4:00"
    assert by_id["UM"].timezone_name is None
    rendered = team_calendar.render_clock(clocks, requester_clock=by_id["UA"])
    assert "Time zone not configured" in rendered
    assert "No locations or time zones were inferred" in rendered


def test_working_hours_status_is_calculated_from_local_time():
    assert team_calendar._availability(
        datetime(2026, 10, 4, 12, 0), time(9), time(18)) == "Working hours"
    assert team_calendar._availability(
        datetime(2026, 10, 4, 8, 30), time(9), time(18)) == "Near working-hours boundary"
    assert team_calendar._availability(
        datetime(2026, 10, 4, 22, 0), time(9), time(18)) == "Outside working hours"


def test_calendar_renderer_produces_real_month_grid_png():
    tasks = [
        task("1", "Late P1", owners=("UA",), priority="P1", due=TODAY - timedelta(days=1)),
        task("2", "Today P2", owners=("UP",), priority="P2", due=TODAY),
        task("3", "Soon P1", owners=("UA",), priority="P1", due=TODAY + timedelta(days=2)),
    ]
    clocks = [
        team_calendar.MemberClock(
            user_id="UA", name="AasthaA", timezone_name="Asia/Kathmandu",
            location="Nepal", local_time=datetime(2026, 10, 4, 15, 45),
            utc_offset="UTC+5:45", working_start=time(9), working_end=time(18),
            availability="Working hours"),
    ]
    png = visual_analytics.render_team_calendar_png(
        tasks, clocks, today=TODAY,
        name_for_user=lambda value: {"UA": "AasthaA", "UP": "Praveen"}[value])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 50_000


def test_team_clock_renderer_produces_visual_artifact(monkeypatch):
    monkeypatch.setenv("TEAM_TIMEZONES_JSON", '{"UA":"Asia/Kathmandu"}')
    monkeypatch.setenv("TEAM_WORKING_HOURS_JSON", '{"UA":"09:00-18:00"}')
    clocks = team_calendar.member_clocks(
        [{"id": "UA", "name": "AasthaA"}],
        datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc))
    png = visual_analytics.render_team_clock_png(clocks, requester_clock=clocks[0])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 20_000
