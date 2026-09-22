from slack_presentation import (TaskRow, created_collection, distribution, task_collection,
                                task_detail, task_field_list, task_line)


def test_single_task_is_one_compact_slack_line():
    rendered = task_line(TaskRow("Deploy API", "Morgan", "2026-09-27", True, "P1", "Pending"), 1)
    assert rendered == "1. *Deploy API* — Morgan · Sep 27 · P1 · Pending"
    assert "**" not in rendered
    assert "\n" not in rendered


def test_collection_preserves_order_and_one_line_per_task():
    rows = [TaskRow(f"Task {index}", "Unassigned", None, True, "P2", "Pending")
            for index in range(1, 31)]
    rendered = task_collection(rows, "Filtered tasks")
    lines = rendered.splitlines()
    assert lines[0] == "*Filtered tasks* · 30 tasks"
    assert len(lines) == 31
    assert lines[1].startswith("1. *Task 1*")
    assert lines[-1].startswith("30. *Task 30*")


def test_pending_and_completed_rows_are_both_visible():
    rendered = task_collection([
        TaskRow("Open", "Alex", None, True, "P2", "Pending"),
        TaskRow("Closed", "Alex", None, True, "P2", "Completed"),
    ])
    assert "*Open*" in rendered and "· Pending" in rendered
    assert "*Closed*" in rendered and "· Completed" in rendered


def test_unreadable_fields_can_be_omitted_without_false_defaults():
    rendered = task_line(TaskRow("Restricted", show_due=False), 1)
    assert rendered == "1. *Restricted*"


def test_analytics_distribution_is_compact_structured_output():
    rendered = distribution("Status distribution", {"Pending": 7, "Completed": 3})
    assert rendered.count("\n") == 2
    assert "• Pending ·" in rendered
    assert "• Completed ·" in rendered


def test_professional_created_task_template_uses_labeled_fields():
    row = TaskRow("Prepare report", "AasthaA", "2026-09-25", True, "P3", "Pending")
    rendered = created_collection([row])
    assert rendered.startswith("*Action Items Created* · 1 task")
    assert task_detail(row) in rendered
    assert "  • Assignee: AasthaA" in rendered
    assert "  • Due: Sep 25" in rendered
    assert rendered.endswith("Action item created successfully and verified in *Action Items*.")


def test_requested_existing_field_template_never_renders_raw_structure():
    rendered = task_field_list(
        TaskRow("Review API", "Praveen", "2026-09-27", True, "No priority", "Pending"))
    assert "• Task: Review API" in rendered
    assert "• Assignee: Praveen" in rendered
    assert "{" not in rendered and "}" not in rendered
