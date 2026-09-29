from datetime import date, timedelta

import pytest

import project_intelligence
import visual_analytics


def task(item_id, name, *, owner=(), priority="P3", due=None, completed=False):
    return project_intelligence.NormalizedTask(
        item_id=item_id, item={}, name=name, owner_ids=tuple(owner),
        priority=priority, due_date=due, completed=completed,
        status="Completed" if completed else "Pending", created_date=None)


def test_workload_chart_data_matches_normalized_snapshot():
    tasks = [task("1", "A", owner=("UA",)), task("2", "B", owner=("UA",)),
             task("3", "C", owner=("UP",)), task("4", "D", completed=True, owner=("UP",)),
             task("5", "E")]
    report = project_intelligence.calculate_workload(
        tasks, {}, lambda value: {"UA": "AasthaA", "UP": "Praveen"}[value],
        date(2026, 9, 25), {"UA", "UP"})
    dataset = visual_analytics.workload_dataset(report)
    assert dataset.series == (("AasthaA", 2), ("Praveen", 1), ("Unassigned", 1))
    assert dataset.scope == "pending tasks" and dataset.record_count == 4


def test_priority_completion_and_deadline_data_use_exact_scopes():
    today = date(2026, 9, 25)
    tasks = [
        task("1", "Late", priority="P1", due=today - timedelta(days=1)),
        task("2", "Today", priority="P2", due=today),
        task("3", "Tomorrow", priority="P3", due=today + timedelta(days=1)),
        task("4", "Done", priority="P1", due=today, completed=True),
    ]
    priority = visual_analytics.build_dataset(tasks, "priority", today=today, name_for_user=str)
    completion = visual_analytics.build_dataset(tasks, "completion", today=today, name_for_user=str)
    deadlines = visual_analytics.build_dataset(tasks, "deadlines", today=today, name_for_user=str)
    assert priority.series == (("P1", 1), ("P2", 1), ("P3", 1))
    assert priority.title == "Pending Task Priority Distribution"
    assert completion.series == (("Pending", 3), ("Completed", 1))
    assert deadlines.series == (("Overdue", 1), ("Due today", 1), ("Next 24h", 1))


def test_svg_is_vector_data_with_exact_labels_and_no_temporary_path():
    dataset = visual_analytics.VisualDataset(
        "priority", "Pending Priority", "pending tasks", (("P1", 4), ("P2", 2)), 6,
        "Six tasks.")
    svg = visual_analytics.render_svg(dataset)
    assert svg.startswith("<svg") and "Pending Priority" in svg
    assert ">4<" in svg and ">2<" in svg
    assert "/tmp" not in svg and "file://" not in svg


def test_response_mode_is_deterministic_and_does_not_force_small_charts():
    assert visual_analytics.choose_response_mode(explicit_visual=True, value_count=2) == "chart"
    assert visual_analytics.choose_response_mode(value_count=1) == "text"
    assert visual_analytics.choose_response_mode(task_level=True, value_count=5) == "table"
    assert visual_analytics.choose_response_mode(dashboard=True, value_count=4) == "dashboard"


def test_part_to_whole_and_real_time_series_have_dedicated_renderers():
    completion = visual_analytics.VisualDataset(
        "completion", "Completion", "all tasks", (("Pending", 3), ("Completed", 2)), 5,
        "Two completed.")
    trend = visual_analytics.VisualDataset(
        "completed_trend", "Completed Over Time", "timestamped tasks",
        (("2026-09-24", 1), ("2026-09-25", 3)), 4, "Four records.")
    assert "stroke-dasharray" in visual_analytics.render_chart(completion)
    assert "polyline" in visual_analytics.render_chart(trend)
    with pytest.raises(ValueError, match="at least two"):
        visual_analytics.render_line_svg(visual_analytics.VisualDataset(
            "completed_trend", "Trend", "tasks", (("2026-09-25", 1),), 1, "One."))


def test_matplotlib_renderers_produce_real_png_images():
    workload = visual_analytics.VisualDataset(
        "workload", "Pending Workload by Owner", "pending tasks",
        (("AasthaA", 14), ("Praveen", 9), ("Unassigned", 2)), 25, "Workload.")
    for chart_type in ("bar", "pie", "table"):
        image = visual_analytics.render_chart_png(workload, chart_type)
        assert image.startswith(b"\x89PNG\r\n\x1a\n") and len(image) > 10_000


def test_dashboard_and_task_table_render_as_png():
    workload = visual_analytics.VisualDataset(
        "workload", "Workload", "pending", (("AasthaA", 3), ("Praveen", 2)), 5, "")
    priority = visual_analytics.VisualDataset(
        "priority", "Priority", "pending", (("P1", 2), ("P2", 3)), 5, "")
    table = visual_analytics.TableDataset(
        "Overdue", "authorized", ("Task", "Owner", "Priority", "Due"),
        (("Client report", "AasthaA", "P1", "Sep 24"),), "One task.")
    assert visual_analytics.render_dashboard_png([workload, priority]).startswith(b"\x89PNG")
    assert visual_analytics.render_table_png(table).startswith(b"\x89PNG")
