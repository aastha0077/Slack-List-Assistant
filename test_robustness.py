import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch

import config
import main
import slack_tools
from intent_parser import parse_intent, _local_parse, _local_parse_create, _local_parse_mutation

TZ = ZoneInfo("Asia/Kathmandu")
TODAY = datetime.now(TZ).date()
TODAY_ISO = TODAY.isoformat()

SCHEMA = {
    "schema": [
        {"id": "col_name", "key": "name", "name": "Task Name", "type": "text"},
        {"id": "col_assignee", "key": "assignee", "name": "Assignee", "type": "user"},
        {"id": "col_due", "key": "due_date", "name": "Due Date", "type": "date"},
        {"id": "col_priority", "key": "priority", "name": "Priority", "type": "select"},
        {"id": "col_completed", "key": "completed", "name": "Completed", "type": "checkbox"},
    ]
}

def make_item(item_id, name, completed=False, user_id="U12345", due="2026-09-20", priority="P2"):
    return {
        "id": item_id,
        "fields": [
            {"column_id": "col_name", "text": name, "value": name},
            {"column_id": "col_assignee", "user": [{"id": user_id}], "value": [user_id]},
            {"column_id": "col_due", "date": due, "value": due},
            {"column_id": "col_priority", "select": {"name": priority}, "value": priority},
            {"column_id": "col_completed", "checkbox": completed, "value": completed},
        ]
    }


class TestPriorityNormalization:
    @pytest.mark.parametrize("input_val,expected", [
        ("P1", "P1"),
        ("p1", "P1"),
        ("priority p1", "P1"),
        ("urgent", "P1"),
        ("critical", "P1"),
        ("highest", "P1"),
        ("high", "P1"),
        ("medium", "P2"),
        ("med", "P2"),
        ("p2", "P2"),
        ("normal", "P3"),
        ("p3", "P3"),
        ("low", "P4"),
        ("lowest", "P4"),
        ("p4", "P4"),
        ("invalid", None),
        ("", None),
        (None, None),
    ])
    def test_normalize_priority(self, input_val, expected):
        assert config.normalize_priority(input_val) == expected


class TestNaturalCompletionPhrases:
    @pytest.mark.parametrize("phrase,expected_name", [
        ("close the login bug", "login bug"),
        ("closed the login bug", "login bug"),
        ("I closed the payment flow", "payment flow"),
        ("done with database migration", "database migration"),
        ("mark documentation as closed", "documentation"),
        ("I wrapped up backend refactoring", "backend refactoring"),
        ("login bug is closed", "login bug"),
    ])
    def test_close_and_wrap_up_phrases(self, phrase, expected_name):
        res = parse_intent(phrase)
        assert res.get("intent") == "complete"
        assert res.get("task_name", "").lower() == expected_name.lower()


class TestNaturalQueries:
    @pytest.mark.parametrize("query,expected_filters", [
        ("anything overdue?", {"intent": "list", "overdue": True}),
        ("what is pending?", {"intent": "list", "status": "open", "completed": False}),
        ("what's left?", {"intent": "list", "status": "open", "completed": False}),
        ("show my work", {"intent": "list", "assignee_self": True}),
        ("what do I need to do?", {"intent": "list", "assignee_self": True}),
        ("what have I completed?", {"intent": "list", "assignee_self": True, "completed": True}),
        ("show everything", {"intent": "list", "all_tasks": True}),
    ])
    def test_natural_read_queries(self, query, expected_filters):
        res = parse_intent(query)
        for k, v in expected_filters.items():
            assert res.get(k) == v, f"Query '{query}' expected {k}={v}, got {res.get(k)}"


class TestSingleCreateMetadataStripping:
    def test_create_with_assignee_due_and_priority(self):
        text = "create login flow for Aastha due Friday priority P1"
        res = parse_intent(text)
        assert res.get("intent") == "create"
        assert res.get("task_name") == "login flow"
        assert res.get("assignee") == "Aastha"
        assert res.get("priority") == "P1"
        assert res.get("due_date") is not None

    def test_create_with_mention_and_tomorrow(self):
        text = "add task Database Optimization for <@U12345|John> due tomorrow p2"
        res = parse_intent(text)
        assert res.get("intent") == "create"
        assert res.get("task_name") == "Database Optimization"
        assert res.get("assignee") == "<@U12345>"
        assert res.get("priority") == "P2"
        assert res.get("due_date") == (TODAY + timedelta(days=1)).isoformat()


class TestCompletionVerification:
    def test_verified_completion(self):
        items_store = [make_item("item_1", "Feature X", completed=False)]

        def mock_complete(item_id, ctx, list_id):
            items_store[0]["fields"][4]["checkbox"] = True
            items_store[0]["fields"][4]["value"] = True
            return True

        with patch.object(slack_tools, "get_list_schema", return_value=SCHEMA), \
             patch.object(slack_tools, "list_action_items", side_effect=lambda ctx, list_id: items_store), \
             patch.object(slack_tools, "complete_action_item", side_effect=mock_complete), \
             patch.object(config, "has_permission", return_value=True):

            ctx = config.build_context(user_id="U12345", channel_id="C_DEV")
            parsed = {"intent": "complete", "task_name": "Feature X"}
            msg = main.handle_mutation(parsed, ctx, "test_mem")
            assert "Action item completed" in msg
            assert "Feature X" in msg
            assert "Completed" in msg

    def test_failed_completion_verification(self):
        # API was called but Slack List still returns completed=False
        items_store = [make_item("item_1", "Feature X", completed=False)]

        def mock_complete_noop(item_id, ctx, list_id):
            return True

        with patch.object(slack_tools, "get_list_schema", return_value=SCHEMA), \
             patch.object(slack_tools, "list_action_items", side_effect=lambda ctx, list_id: items_store), \
             patch.object(slack_tools, "complete_action_item", side_effect=mock_complete_noop), \
             patch.object(config, "has_permission", return_value=True):

            ctx = config.build_context(user_id="U12345", channel_id="C_DEV")
            parsed = {"intent": "complete", "task_name": "Feature X"}
            msg = main.handle_mutation(parsed, ctx, "test_mem")
            assert "still shows pending" in msg


class TestStaleThreadContext:
    def test_stale_item_filtered_out(self):
        item_1 = make_item("item_1", "Old Task")
        item_2 = make_item("item_2", "Current Task")
        main.store_view("stale_mem", [item_1, item_2])

        current_items = [item_2]
        parsed = {"intent": "complete", "selection": "first", "selection_index": 1}
        ctx = config.build_context(user_id="U12345", channel_id="C_DEV")

        targets = main.resolve_targets(parsed, current_items, SCHEMA, "stale_mem", ctx=ctx, intent="complete")
        assert len(targets) == 1
        assert targets[0]["id"] == "item_2"


class TestUnassignedCreation:
    def test_create_without_assignee(self):
        created_items = []
        def mock_create(name, priority, assignee, due_date, ctx, list_id):
            it = make_item("new_1", name, completed=False, user_id=assignee, priority=priority or "P3", due=due_date or "2026-09-25")
            created_items.append(it)
            return it

        with patch.object(slack_tools, "get_list_schema", return_value=SCHEMA), \
             patch.object(slack_tools, "list_action_items", return_value=[]), \
             patch.object(slack_tools, "create_action_item", side_effect=mock_create), \
             patch.object(config, "has_permission", return_value=True):

            ctx = config.build_context(user_id="U12345", channel_id="C_DEV")
            parsed = {"intent": "create", "task_name": "Documentation Update"}
            msg = main.handle_create(parsed, ctx)
            assert "Action item created successfully" in msg
            assert "Documentation Update" in msg
            assert len(created_items) == 1
            assert created_items[0]["fields"][1]["user"][0]["id"] is None


class TestPriorityTwoParsingDetails:
    def test_prepare_project_report_to_user(self):
        text = "add task Prepare project report to @AasthaA due 2028-09-09 p2"
        res = parse_intent(text)
        assert res.get("intent") == "create"
        assert res.get("task_name") == "Prepare project report"
        assert res.get("assignee") == "AasthaA"
        assert res.get("due_date") == "2028-09-09"
        assert res.get("priority") == "P2"

    def test_complete_python_assignment_to_me(self):
        text = "add task Complete Python assignment to me"
        res = parse_intent(text)
        assert res.get("intent") == "create"
        assert res.get("task_name") == "Complete Python assignment"
        assert res.get("assignee_self") is True


class TestBulkOperationsAndThreadIsolation:
    def test_complete_all_in_thread(self):
        main._pending.clear()
        items_store = [
            make_item("item_1", "Task Alpha", completed=False),
            make_item("item_2", "Task Beta", completed=False),
        ]

        def mock_complete(item_id, ctx, list_id):
            for it in items_store:
                if it["id"] == item_id:
                    it["fields"][4]["checkbox"] = True
                    it["fields"][4]["value"] = True
            return True

        with patch.object(slack_tools, "get_list_schema", return_value=SCHEMA), \
             patch.object(slack_tools, "list_action_items", side_effect=lambda ctx, list_id: [dict(x) for x in items_store]), \
             patch.object(slack_tools, "complete_action_item", side_effect=mock_complete), \
             patch.object(config, "has_permission", return_value=True):

            user_id = "U12345"
            channel_id = "C_DEV"
            thread_ts = "1726000000.123456"

            # Show tasks
            main.process("show tasks", user_id, channel_id, thread_ts)

            # "I completed all"
            res = main.process("I completed all", user_id, channel_id, thread_ts)
            assert "Task Alpha" in res
            assert "Task Beta" in res
            assert all(slack_tools.extract_completed(x, SCHEMA) for x in items_store)

    def test_positional_without_context_raises_clarification(self):
        main._pending.clear()
        items_store = [make_item("item_1", "Lone Task", completed=False)]

        with patch.object(slack_tools, "get_list_schema", return_value=SCHEMA), \
             patch.object(slack_tools, "list_action_items", return_value=items_store), \
             patch.object(config, "has_permission", return_value=True):

            # No "show tasks" executed in this thread
            res = main.process("I completed the last", "U12345", "C_DEV", "thread_unseen_999")
            assert "recently displayed action items in this thread" in res or "couldn't find" in res

