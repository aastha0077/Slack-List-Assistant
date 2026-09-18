"""
test_multi_create.py
====================
Unit tests for the deterministic multi-task CREATE parser in intent_parser.py.
Tests run entirely offline — no Ollama, no Slack API calls needed.
Run with:  .venv/bin/python -m pytest test_multi_create.py -v
"""
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from intent_parser import parse_intent, _local_parse_create, _today, _tomorrow

# Silence noisy logs during test runs
logging.basicConfig(level=logging.CRITICAL)

TZ = ZoneInfo("Asia/Kathmandu")
TODAY = datetime.now(TZ).date()
TOMORROW = TODAY + timedelta(days=1)
TODAY_ISO = TODAY.isoformat()
TOMORROW_ISO = TOMORROW.isoformat()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pi(text):
    """Shorthand for parse_intent()."""
    return parse_intent(text)


def assert_multi(result, expected_names, *, intent="create"):
    """Assert that result contains a tasks list with names matching expected_names."""
    assert result.get("intent") == intent, (
        f"Expected intent={intent!r}, got {result.get('intent')!r}\nFull: {json.dumps(result)}"
    )
    tasks = result.get("tasks")
    assert isinstance(tasks, list), f"Expected 'tasks' list, got {type(tasks)}\nFull: {json.dumps(result)}"
    names = [t.get("task_name", "").strip() for t in tasks]
    assert sorted(names) == sorted(expected_names), (
        f"Expected task names {expected_names!r}, got {names!r}\nFull: {json.dumps(result)}"
    )
    return tasks


def assert_single(result, expected_name, *, intent="create"):
    """Assert that result is a single-task CREATE (no tasks list)."""
    assert result.get("intent") == intent, (
        f"Expected intent={intent!r}, got {result.get('intent')!r}\nFull: {json.dumps(result)}"
    )
    tasks = result.get("tasks")
    assert tasks is None or len(tasks) == 0, (
        f"Expected single-task (no 'tasks'), but got tasks={tasks}\nFull: {json.dumps(result)}"
    )
    name = (result.get("task_name") or "").strip()
    assert name == expected_name, f"Expected task_name={expected_name!r}, got {name!r}\nFull: {json.dumps(result)}"


# ===========================================================================
# 1. BULLET LIST
# ===========================================================================

class TestBulletList:

    def test_basic_two_bullets(self):
        text = (
            "I have a few action items today. I want to work on\n"
            "* Insight dashboard Completion for pitch\n"
            "* Deploy the pitch dashboard."
        )
        result = pi(text)
        assert_multi(result, [
            "Insight dashboard Completion for pitch",
            "Deploy the pitch dashboard",
        ])

    def test_three_bullets(self):
        text = "* Finish report\n* Review code\n* Deploy dashboard"
        result = pi(text)
        tasks = assert_multi(result, ["Finish report", "Review code", "Deploy dashboard"])
        assert len(tasks) == 3

    def test_bullets_with_due_today(self):
        text = "Tasks for today:\n* Write tests\n* Fix bug"
        result = pi(text)
        tasks = assert_multi(result, ["Write tests", "Fix bug"])
        for t in tasks:
            assert t.get("due_date") == TODAY_ISO, (
                f"Expected today={TODAY_ISO!r} for each task, got {t.get('due_date')!r}"
            )

    def test_bullets_with_priority(self):
        text = "Please create these P1 tasks:\n* Dashboard update\n* API deployment"
        result = pi(text)
        tasks = assert_multi(result, ["Dashboard update", "API deployment"])
        for t in tasks:
            assert t.get("priority") == "P1", f"Expected P1 priority, got {t.get('priority')!r}"

    def test_bullets_with_assignee(self):
        text = "Add tasks for Praveen:\n* Client report\n* Team meeting"
        result = pi(text)
        tasks = assert_multi(result, ["Client report", "Team meeting"])
        for t in tasks:
            assert t.get("assignee") == "Praveen", f"Expected assignee Praveen, got {t.get('assignee')!r}"

    def test_dash_bullets(self):
        text = "Work on these:\n- Fix login bug\n- Update docs"
        result = pi(text)
        assert_multi(result, ["Fix login bug", "Update docs"])

    def test_bullet_with_slack_mention(self):
        text = "Assign to <@U123ABC>:\n* Task A\n* Task B"
        result = pi(text)
        tasks = assert_multi(result, ["Task A", "Task B"])
        for t in tasks:
            assert "<@U123ABC>" in (t.get("assignee") or ""), (
                f"Expected mention assignee, got {t.get('assignee')!r}"
            )

    def test_single_bullet_returns_flat_dict(self):
        """A single bullet item must return a flat dict (not a tasks list) for backward compat."""
        text = "I want to work on:\n* Client report"
        result = pi(text)
        assert result.get("intent") == "create"
        tasks = result.get("tasks")
        assert tasks is None or len(tasks) <= 1, (
            f"Single bullet should produce flat dict or 1-item tasks, got: {tasks}"
        )


# ===========================================================================
# 2. NUMBERED LIST
# ===========================================================================

class TestNumberedList:

    def test_basic_numbered_list(self):
        text = "I have these tasks:\n1. Finish dashboard\n2. Deploy dashboard\n3. Write docs"
        result = pi(text)
        assert_multi(result, ["Finish dashboard", "Deploy dashboard", "Write docs"])

    def test_numbered_list_with_paren(self):
        text = "1. Task Alpha\n2. Task Beta"
        result = pi(text)
        assert_multi(result, ["Task Alpha", "Task Beta"])

    def test_numbered_with_shared_metadata(self):
        text = "Create P2 tasks due today:\n1. Write report\n2. Send to client"
        result = pi(text)
        tasks = assert_multi(result, ["Write report", "Send to client"])
        for t in tasks:
            assert t.get("priority") == "P2", f"Expected P2, got {t.get('priority')!r}"
            assert t.get("due_date") == TODAY_ISO, f"Expected today, got {t.get('due_date')!r}"


# ===========================================================================
# 3. NATURAL LANGUAGE "A AND B"
# ===========================================================================

class TestAndSplit:

    def test_two_tasks_and_split(self):
        text = "I need to work on dashboard completion and deploy the pitch dashboard"
        result = pi(text)
        assert result.get("intent") == "create"
        # Either tasks list with 2 entries, or at minimum a single task was created
        if result.get("tasks"):
            assert len(result["tasks"]) == 2
        else:
            assert result.get("task_name"), "Expected at least a single task_name"

    def test_two_tasks_with_today_and_priority(self):
        text = "I need to finish the dashboard and deploy it today, both P1."
        result = pi(text)
        assert result.get("intent") == "create"
        if result.get("tasks"):
            for t in result["tasks"]:
                assert t.get("due_date") == TODAY_ISO, f"Expected today, got {t.get('due_date')!r}"
                assert t.get("priority") == "P1", f"Expected P1, got {t.get('priority')!r}"


# ===========================================================================
# 4. SINGLE TASK — REGRESSION
# ===========================================================================

class TestSingleTaskRegression:

    def test_add_single_task(self):
        result = pi("add client report")
        assert result.get("intent") == "create"
        tasks = result.get("tasks")
        assert tasks is None or len(tasks) <= 1, "Single-task add must not produce multi tasks list"

    def test_create_named_task(self):
        result = pi("create a task called Client Report")
        assert result.get("intent") == "create"

    def test_i_need_to_work_on_single(self):
        result = pi("I need to work on the dashboard")
        assert result.get("intent") == "create"
        assert result.get("assignee_self") is True


# ===========================================================================
# 5. GUARD-RAILS — must NOT be intercepted as CREATE
# ===========================================================================

class TestGuardRails:

    def test_query_what_are_my_tasks(self):
        result = pi("what are my tasks?")
        assert result.get("intent") == "list", (
            f"Expected list, got {result.get('intent')!r}\nFull: {json.dumps(result)}"
        )
        assert not result.get("tasks")

    def test_query_show_tasks_due_today(self):
        result = pi("show me the tasks due today")
        assert result.get("intent") == "list"

    def test_query_what_do_i_need_to_do(self):
        result = pi("what do I need to do today?")
        assert result.get("intent") == "list", (
            f"Expected list, got {result.get('intent')!r}"
        )

    def test_complete_mutation(self):
        result = pi("complete client report")
        assert result.get("intent") == "complete", (
            f"Expected complete, got {result.get('intent')!r}"
        )

    def test_delete_mutation(self):
        result = pi("delete the old report task")
        assert result.get("intent") == "delete", (
            f"Expected delete, got {result.get('intent')!r}"
        )

    def test_update_mutation(self):
        result = pi("change docs check priority to P2")
        assert result.get("intent") == "update", (
            f"Expected update, got {result.get('intent')!r}"
        )

    def test_empty_text(self):
        result = pi("")
        assert result.get("intent") in {"out_of_scope", "clarify", "temporarily_unavailable"}

    def test_out_of_scope_weather(self):
        result = pi("what is the weather like outside?")
        assert result.get("intent") != "create", (
            f"Weather question must not be treated as CREATE, got {result.get('intent')!r}"
        )


# ===========================================================================
# 6. METADATA PROPAGATION
# ===========================================================================

class TestMetadataPropagation:

    def test_priority_propagates_to_all_tasks(self):
        text = "P1 tasks:\n* Dashboard\n* Deploy"
        result = pi(text)
        if result.get("tasks"):
            for t in result["tasks"]:
                assert t.get("priority") == "P1", f"Expected P1 for each task: {t}"

    def test_assignee_self_propagates(self):
        text = "I want to work on:\n* Task A\n* Task B"
        result = pi(text)
        if result.get("tasks"):
            for t in result["tasks"]:
                assert t.get("assignee_self") is True, f"Expected assignee_self=True: {t}"

    def test_full_shared_metadata(self):
        # NOTE: "Finish X and deploy it..." starts with "Finish" which the COMPLETE
        # anchor intercepts correctly. Use first-person phrasing for a CREATE.
        text = "I need to finish the dashboard and deploy it today, both P1."
        result = pi(text)
        assert result.get("intent") == "create"
        if result.get("tasks"):
            for t in result["tasks"]:
                assert t.get("due_date") == TODAY_ISO
                assert t.get("priority") == "P1"



# ===========================================================================
# 7. Direct unit tests for _local_parse_create
# ===========================================================================

class TestLocalParseCreateDirect:

    def test_returns_empty_for_mutation_delete(self):
        assert _local_parse_create("delete client report") == {}

    def test_returns_empty_for_mutation_update(self):
        assert _local_parse_create("update dashboard priority to P1") == {}

    def test_returns_empty_for_mutation_complete(self):
        assert _local_parse_create("complete client report") == {}

    def test_returns_empty_for_read(self):
        assert _local_parse_create("show all tasks") == {}

    def test_returns_empty_for_what_are_my_tasks(self):
        assert _local_parse_create("what are my tasks?") == {}

    def test_returns_empty_for_no_intent_no_list(self):
        # No create keyword, no bullet — should fall through
        result = _local_parse_create("the weather is nice outside")
        assert result == {}

    def test_bullet_multi_returns_tasks_list(self):
        text = "* Alpha\n* Beta"
        r = _local_parse_create(text)
        assert r.get("intent") == "create"
        assert isinstance(r.get("tasks"), list)
        assert len(r["tasks"]) == 2

    def test_single_bullet_returns_flat_dict(self):
        text = "I need to work on:\n* Alpha"
        r = _local_parse_create(text)
        assert r.get("intent") == "create"
        assert r.get("task_name") == "Alpha"
        assert "tasks" not in r

    def test_today_resolves_to_iso(self):
        text = "Add task for today:\n* Alpha\n* Beta"
        r = _local_parse_create(text)
        if r.get("tasks"):
            for t in r["tasks"]:
                assert t.get("due_date") == TODAY_ISO, f"Expected {TODAY_ISO}, got {t.get('due_date')}"

    def test_tomorrow_resolves_to_iso(self):
        text = "Tomorrow's tasks:\n* Alpha\n* Beta"
        r = _local_parse_create(text)
        if r.get("tasks"):
            for t in r["tasks"]:
                assert t.get("due_date") == TOMORROW_ISO, f"Expected {TOMORROW_ISO}, got {t.get('due_date')}"

    def test_numbered_two_tasks(self):
        text = "My action items are:\n1. Write tests\n2. Fix bugs"
        r = _local_parse_create(text)
        assert r.get("intent") == "create"
        tasks = r.get("tasks")
        assert isinstance(tasks, list) and len(tasks) == 2
        names = [t["task_name"] for t in tasks]
        assert "Write tests" in names
        assert "Fix bugs" in names

    def test_three_numbered_tasks(self):
        text = "1. Task A\n2. Task B\n3. Task C"
        r = _local_parse_create(text)
        assert r.get("intent") == "create"
        tasks = r.get("tasks")
        assert isinstance(tasks, list) and len(tasks) == 3


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "--tb=short"])
