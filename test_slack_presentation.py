from datetime import date

from slack_presentation import (
    PresentationStrategy, ResponseComplexity, TaskRow,
    assignee_clarification, clarification, created_collection,
    distribution, empty_state, failure, focus_reason, permission_denied,
    render_slack_table, render_task_card,
    task_collection, task_conflict, task_field_list, task_line,
    task_collection_strategy, validate_slack_response,
)


TODAY = date(2026, 9, 22)


def test_adaptive_task_strategy_uses_tables_for_comparable_task_sets():
    assert task_collection_strategy(0) == (
        ResponseComplexity.SIMPLE, PresentationStrategy.EMPTY_STATE)
    assert task_collection_strategy(3) == (
        ResponseComplexity.STRUCTURED, PresentationStrategy.TASK_TABLE)
    assert task_collection_strategy(12) == (
        ResponseComplexity.STRUCTURED, PresentationStrategy.TASK_TABLE)


def test_shared_task_card_makes_name_dominant_and_detail_secondary():
    rendered = render_task_card(
        "Follow-up Test Task", ("P1", "AasthaA", "Overdue"),
        prefix="1.", detail=focus_reason("P1 + overdue by 4 days"))
    assert rendered == (
        "1. *Follow-up Test Task*\n"
        "   P1 · AasthaA · Overdue\n"
        "   ↳ P1 priority + 4 days overdue")


def test_single_task_is_one_compact_slack_line_in_scan_order():
    rendered = task_line(
        TaskRow("Deploy API", "Morgan", "2026-09-27", True, "P1", "Pending"),
        1, today=TODAY)
    assert rendered == "1. *Deploy API* · P1 · Morgan · Sep 27, 2026 · Pending"
    assert "**" not in rendered and "\n" not in rendered


def test_small_task_collection_uses_slack_safe_table():
    rendered = task_collection([
        TaskRow("Alpha", "Alex", "2026-09-25", True, "P1", "Pending"),
        TaskRow("Beta", "Morgan", "2026-09-26", True, "P2", "Pending"),
    ], "Pending Action Items", today=TODAY)
    assert rendered.startswith("*📋 Pending Action Items*\n\n*2 pending*\n\n```")
    assert "Task   Priority  Owner" in rendered
    assert "Alpha  P1        Alex" in rendered
    assert "Beta   P2        Morgan" in rendered
    assert "Status" not in rendered


def test_large_collection_preserves_order_in_compact_table():
    rows = [TaskRow(f"Task {index}", "Unassigned", None, True, "P2", "Pending")
            for index in range(1, 31)]
    rendered = task_collection(rows, "Filtered tasks", today=TODAY)
    lines = rendered.splitlines()
    assert lines[0] == "*📋 Filtered tasks*"
    assert lines[5].startswith("Task")
    assert lines[7].startswith("Task 1 ")
    assert lines[-2].startswith("Task 30")


def test_table_truncates_long_task_names_for_mobile_width():
    full_name = "Prepare the internship demo checklist with every final verification detail"
    rendered = task_collection([
        TaskRow(full_name, "AasthaA", "2026-09-30", True, "P2", "Pending")
    ], "Action Items", today=TODAY)
    assert "Prepare the internship demo" in rendered
    assert "…" in rendered and "```" in rendered
    assert max(len(line) for line in rendered.splitlines()) <= 78


def test_my_tasks_omit_redundant_owner_but_keep_priority_and_date():
    rendered = task_collection([
        TaskRow("Prepare report", "AasthaA", "2026-09-30", True, "P1", "Pending")
    ], "Your Pending Tasks", today=TODAY)
    assert "AasthaA" not in rendered
    assert "Task            Priority  Due" in rendered
    assert "Prepare report  P1        Sep 30, 2026" in rendered
    assert "Owner" not in rendered
    assert "*1 pending*" in rendered


def test_smart_due_states_and_missing_priority_are_compact():
    overdue = task_line(TaskRow(
        "Late", "Alex", "2026-09-21", True, "No priority", "Pending"), today=TODAY)
    due_today = task_line(TaskRow(
        "Today", "Alex", "2026-09-22", True, "P2", "Pending"), today=TODAY)
    tomorrow = task_line(TaskRow(
        "Tomorrow", "Alex", "2026-09-23", True, "P3", "Pending"), today=TODAY)
    no_due = task_line(TaskRow(
        "Someday", "Alex", None, True, "No priority", "Pending"), today=TODAY)
    assert "— · Alex · 🔴 Overdue (Sep 21, 2026)" in overdue
    assert "🟡 Due today" in due_today and "· Pending" not in due_today
    assert "Due tomorrow" in tomorrow and "· Pending" not in tomorrow
    assert "— · Alex · No due date · Pending" in no_due


def test_completed_tasks_use_checkmark_without_status_column():
    rendered = task_collection([
        TaskRow("Open", "Alex", None, True, "P2", "Pending"),
        TaskRow("Closed", "Alex", None, True, "P2", "Completed"),
    ], today=TODAY)
    assert "Open      P2" in rendered
    assert "Closed ✓  P2" in rendered
    assert "Status" not in rendered
    assert "Completed" not in rendered


def test_all_tasks_group_when_due_attention_improves_scanning():
    rendered = task_collection([
        TaskRow("Late", "Alex", "2026-09-20", True, "P1", "Pending"),
        TaskRow("Today", "Morgan", "2026-09-22", True, "P2", "Pending"),
        TaskRow("Later", "Alex", "2026-09-30", True, "P3", "Pending"),
        TaskRow("Done", "Morgan", "2026-09-21", True, "P2", "Completed"),
    ], group_due=True, group_status=True, today=TODAY)
    assert "*Overdue*" in rendered
    assert "*Due Today*" in rendered
    assert "*Upcoming*" in rendered
    assert "*Completed*" in rendered


def test_useful_collection_summary_is_compact():
    rendered = task_collection([
        TaskRow("A", due_date="2026-09-20", priority="P1", status="Pending"),
        TaskRow("B", due_date="2026-09-22", priority="P1", status="Pending"),
        TaskRow("C", due_date="2026-09-30", priority="P2", status="Pending"),
        TaskRow("D", due_date="2026-10-01", priority="P3", status="Pending"),
    ], today=TODAY)
    assert "*4 pending · 1 overdue · 1 due today*" in rendered


def test_unreadable_fields_can_be_omitted_without_false_defaults():
    rendered = task_line(TaskRow("Restricted", show_due=False), 1, today=TODAY)
    assert rendered == "1. *Restricted*"


def test_created_task_template_is_compact_and_verified():
    row = TaskRow("Prepare report", "AasthaA", "2026-09-25", True, "P3", "Pending")
    rendered = created_collection([row], today=TODAY)
    assert rendered.startswith("*✓ Action Item Created*\n\n*Prepare report*")
    assert "P3 · AasthaA · Sep 25, 2026 · Pending" in rendered
    assert rendered.endswith("Task created successfully and verified in *Action Items*.")


def test_duplicate_conflict_is_compact_and_preserves_requested_and_existing_values():
    requested = TaskRow("Review API", "Praveen", "2026-10-05", True, "P2", "Pending")
    existing = TaskRow("Review API", "Praveen", "2026-09-25", True, "P1", "Pending")
    rendered = task_conflict(requested, existing, today=TODAY)
    assert rendered.startswith("*↔ Existing task differs*")
    assert "Requested: P2 · Oct 5, 2026" in rendered
    assert "Existing: P1 · Sep 25, 2026" in rendered
    assert "No changes made" in rendered


def test_member_clarification_is_concise_and_never_exposes_an_id():
    rendered = assignee_clarification(
        TaskRow("Client presentation", due_date="2026-09-30"), "Asta", today=TODAY)
    assert rendered.startswith("*⚠ Assignee unclear*")
    assert "*Client presentation* · Sep 30, 2026" in rendered
    assert 'match "Asta"' in rendered
    assert "U0C2F3CFQ00" not in rendered


def test_context_aware_empty_state_is_used_verbatim():
    rendered = task_collection(
        [], "Your Pending Tasks", empty_message="You have no pending tasks.", today=TODAY)
    assert rendered == "*📋 My Action Items*\n\nYou have no pending tasks."


def test_slack_table_is_aligned_bounded_and_uses_no_markdown_pipes():
    rendered = render_slack_table(
        ("Task", "Priority", "Owner", "Due"),
        [("A very long task name that must be safely shortened for mobile", "P1", "AasthaA", "Oct 3, 2026")],
        title="Action Items", summary="1 task")
    assert rendered.startswith("*Action Items*\n\n*1 task*\n\n```")
    assert "|" not in rendered
    assert "…" in rendered
    assert max(len(line) for line in rendered.splitlines()) <= 78


def test_collection_summary_counts_exactly_the_rendered_dataset():
    rows = [
        TaskRow("One", priority="P1", due_date="2026-09-20", status="Pending"),
        TaskRow("Two", priority="P1", due_date="2026-09-22", status="Pending"),
        TaskRow("Three", priority="P2", due_date="2026-10-02", status="Pending"),
    ]
    rendered = task_collection(rows, "Action Items", today=TODAY)
    assert "*3 pending · 1 overdue · 1 due today*" in rendered
    table_rows = [line for line in rendered.splitlines()
                  if line.startswith(("One", "Two", "Three"))]
    assert len(table_rows) == len(rows)
    assert "2914 P1" not in rendered


def test_analytics_distribution_is_compact_structured_output():
    rendered = distribution("Status distribution", {"Pending": 7, "Completed": 3})
    assert rendered.startswith("*Status distribution*\n\n```")
    assert "Metric" in rendered and "Count" in rendered
    assert "Pending" in rendered and "Completed" in rendered


def test_labeled_compatibility_view_never_renders_raw_structure():
    rendered = task_field_list(
        TaskRow("Review API", "Praveen", "2026-09-27", True, "No priority", "Pending"))
    assert "• Task: Review API" in rendered
    assert "• Priority: —" in rendered
    assert "{" not in rendered and "}" not in rendered


def test_central_response_states_are_slack_native_structured_and_actionable():
    permission = permission_denied("Your role cannot update this action item.")
    missing = clarification("Please name the action item.")
    failed = failure(
        "I couldn't verify the requested change.",
        next_step="Check the Action Items list before retrying.")
    empty = empty_state(
        "Focus Today", "No action items are due today.",
        context="You still have 5 pending tasks.")

    assert permission.startswith("*Permission denied*\n\n")
    assert "*Next step:*" in permission
    assert missing == "*More information needed*\n\nPlease name the action item."
    assert failed.startswith("*Action not completed*\n\n")
    assert empty == ("*Focus Today*\n\nNo action items are due today.\n\n"
                     "You still have 5 pending tasks.")
    for rendered in (permission, missing, failed, empty):
        assert "**" not in rendered
        assert "{" not in rendered and "}" not in rendered
        assert "U0C2F3CFQ00" not in rendered


def test_final_quality_gate_normalizes_safe_slack_presentation_defects():
    rendered, issues = validate_slack_response(
        "**Action Items**\n\nOwner: U0C2F3CFQ00\nList: F0C1GLU2JVB\n"
        "[:red_circle:](https://a.slack-edge.com/icon.png) High",
        resolve_user=lambda user_id: "AasthaA")
    assert rendered.startswith("*Action Items*")
    assert "Owner: AasthaA" in rendered and "List: Action Items" in rendered
    assert "🔴 High" in rendered
    assert "**" not in rendered and "slack-edge.com" not in rendered
    assert "U0C2F3CFQ00" not in rendered and "F0C1GLU2JVB" not in rendered
    assert set(issues) == {
        "markdown_bold", "emoji_image_link", "raw_user_id", "raw_list_id"}


def test_final_quality_gate_blocks_internal_debug_and_raw_json():
    traceback, traceback_issues = validate_slack_response(
        "Traceback (most recent call last):\nKeyError: 'assignee'")
    raw_json, json_issues = validate_slack_response('{"ok": false, "error": "internal"}')
    for rendered in (traceback, raw_json):
        assert rendered.startswith("*Action not completed*")
        assert "Traceback" not in rendered and "KeyError" not in rendered
        assert "{\"" not in rendered
    assert traceback_issues == ("internal_debug_output",)
    assert json_issues == ("raw_json",)


def test_quality_gate_reports_structural_issues_without_changing_business_content():
    rendered, issues = validate_slack_response(
        "*Status*\nPending work\n\n*Status*\nNo action items found. 3 pending tasks\n```data")
    assert rendered.endswith("\n```")
    assert "Pending work" in rendered
    assert set(issues) == {
        "unclosed_code_fence", "duplicate_section", "contradictory_empty_state"}
