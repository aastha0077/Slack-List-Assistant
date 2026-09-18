import pytest
from unittest.mock import patch
import main
import slack_tools
import config

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


@pytest.fixture(autouse=True)
def clean_pending():
    main._pending.clear()
    yield
    main._pending.clear()


def test_regression_exact_item_id_order_and_last_resolution():
    """
    Regression test requested by user:
    1. Bot displays three tasks.
    2. Context stores their exact item IDs in order.
    3. User replies 'I completed the last' (in the same thread or replying to the root bot message).
    4. The third item's ID is selected.
    5. Only the third item is completed.
    """
    items_store = [
        make_item("item_A", "Prepare project report", completed=False),
        make_item("item_B", "Complete Python assignment", completed=False),
        make_item("item_C", "Deploy production release", completed=False),
    ]

    def mock_list_action_items(ctx, list_id):
        return [dict(x) for x in items_store]

    def mock_complete_action_item(item_id, ctx, list_id):
        for it in items_store:
            if it["id"] == item_id:
                for f in it["fields"]:
                    if f.get("column_id") == "col_completed":
                        f["checkbox"] = True
                        f["value"] = True
                return True
        return False

    with patch.object(slack_tools, "get_list_schema", return_value=SCHEMA), \
         patch.object(slack_tools, "list_action_items", side_effect=mock_list_action_items), \
         patch.object(slack_tools, "complete_action_item", side_effect=mock_complete_action_item), \
         patch.object(config, "has_permission", return_value=True):

        user_id = "U_STUDENT"
        channel_id = "C_DEV"
        root_msg_ts = "1726000000.000100"
        bot_response_ts = "1726000000.000200"

        # 1. User asks 'show tasks' at channel root (thread_ts=None, msg_ts=root_msg_ts)
        res_display = main.process("show tasks", user_id, channel_id, thread_ts=None, msg_ts=root_msg_ts)
        assert "Prepare project report" in res_display
        assert "Complete Python assignment" in res_display
        assert "Deploy production release" in res_display

        # Simulate bot message post and timestamp recording
        main.record_bot_response(channel_id, bot_response_ts, thread_ts=None, msg_ts=root_msg_ts, user_id=user_id)

        # 2. Verify stored displayed tasks have exact item IDs in order
        ctx_data = main.get_thread_context(channel_id, thread_ts=bot_response_ts, msg_ts="1726000005.000300", user_id=user_id)
        assert ctx_data is not None
        displayed = ctx_data["displayed_tasks"]
        assert len(displayed) == 3
        assert displayed[0]["item_id"] == "item_A" and displayed[0]["name"] == "Prepare project report"
        assert displayed[1]["item_id"] == "item_B" and displayed[1]["name"] == "Complete Python assignment"
        assert displayed[2]["item_id"] == "item_C" and displayed[2]["name"] == "Deploy production release"

        # 3. User replies 'I completed the last' in the thread under the bot's response message
        reply_ts = "1726000005.000300"
        res_complete = main.process("I completed the last", user_id, channel_id, thread_ts=bot_response_ts, msg_ts=reply_ts)

        # 4. Third item is selected and confirmed
        assert "Deploy production release" in res_complete
        assert "Prepare project report" not in res_complete
        assert "Complete Python assignment" not in res_complete

        # 5. ONLY the third item is completed
        assert slack_tools.extract_completed(items_store[0], SCHEMA) is False
        assert slack_tools.extract_completed(items_store[1], SCHEMA) is False
        assert slack_tools.extract_completed(items_store[2], SCHEMA) is True


def test_positional_references_all_variants():
    """
    Test first, second, third, last, both, all in threaded flows.
    """
    items_store = [
        make_item("item_1", "Task One", completed=False),
        make_item("item_2", "Task Two", completed=False),
        make_item("item_3", "Task Three", completed=False),
    ]

    def mock_list_action_items(ctx, list_id):
        return [dict(x) for x in items_store]

    def mock_complete_action_item(item_id, ctx, list_id):
        for it in items_store:
            if it["id"] == item_id:
                for f in it["fields"]:
                    if f.get("column_id") == "col_completed":
                        f["checkbox"] = True
                        f["value"] = True
                return True
        return False

    with patch.object(slack_tools, "get_list_schema", return_value=SCHEMA), \
         patch.object(slack_tools, "list_action_items", side_effect=mock_list_action_items), \
         patch.object(slack_tools, "complete_action_item", side_effect=mock_complete_action_item), \
         patch.object(config, "has_permission", return_value=True):

        user_id = "U123"
        channel_id = "C_TEST"
        thread_ts = "1726000000.000999"

        # Show 3 tasks
        main.process("show tasks", user_id, channel_id, thread_ts=thread_ts)

        # Test 'first'
        res_first = main.process("complete first", user_id, channel_id, thread_ts=thread_ts)
        assert "Task One" in res_first
        assert slack_tools.extract_completed(items_store[0], SCHEMA) is True

        # Test 'second'
        res_second = main.process("complete the second", user_id, channel_id, thread_ts=thread_ts)
        assert "Task Two" in res_second
        assert slack_tools.extract_completed(items_store[1], SCHEMA) is True

        # Test 'third'
        res_third = main.process("complete the third", user_id, channel_id, thread_ts=thread_ts)
        assert "Task Three" in res_third
        assert slack_tools.extract_completed(items_store[2], SCHEMA) is True

        # Reset completed status
        for it in items_store:
            for f in it["fields"]:
                if f.get("column_id") == "col_completed":
                    f["checkbox"] = False
                    f["value"] = False

        # Show 3 tasks again
        main.process("show tasks", user_id, channel_id, thread_ts=thread_ts)

        # Test 'all'
        res_all = main.process("complete all", user_id, channel_id, thread_ts=thread_ts)
        assert "Task One" in res_all
        assert "Task Two" in res_all
        assert "Task Three" in res_all
        assert all(slack_tools.extract_completed(it, SCHEMA) for it in items_store)


def test_both_variant_with_two_items():
    """
    Test 'complete both' when exactly two tasks are displayed.
    """
    items_store = [
        make_item("item_1", "Task Alpha", completed=False),
        make_item("item_2", "Task Beta", completed=False),
    ]

    def mock_list_action_items(ctx, list_id):
        return [dict(x) for x in items_store]

    def mock_complete_action_item(item_id, ctx, list_id):
        for it in items_store:
            if it["id"] == item_id:
                for f in it["fields"]:
                    if f.get("column_id") == "col_completed":
                        f["checkbox"] = True
                        f["value"] = True
                return True
        return False

    with patch.object(slack_tools, "get_list_schema", return_value=SCHEMA), \
         patch.object(slack_tools, "list_action_items", side_effect=mock_list_action_items), \
         patch.object(slack_tools, "complete_action_item", side_effect=mock_complete_action_item), \
         patch.object(config, "has_permission", return_value=True):

        user_id = "U123"
        channel_id = "C_TEST"
        thread_ts = "1726000000.000888"

        main.process("show tasks", user_id, channel_id, thread_ts=thread_ts)
        res_both = main.process("complete both", user_id, channel_id, thread_ts=thread_ts)
        assert "Task Alpha" in res_both
        assert "Task Beta" in res_both
        assert all(slack_tools.extract_completed(it, SCHEMA) for it in items_store)


def test_both_clarification_on_three_items():
    """Test 'both' prompts for clarification when more than 2 items exist."""
    items_store = [
        make_item("item_1", "Alpha Task", completed=False),
        make_item("item_2", "Beta Task", completed=False),
        make_item("item_3", "Gamma Task", completed=False),
    ]

    with patch.object(slack_tools, "get_list_schema", return_value=SCHEMA), \
         patch.object(slack_tools, "list_action_items", return_value=items_store), \
         patch.object(config, "has_permission", return_value=True):

        user_id = "U12345"
        channel_id = "C_TEST"
        thread_ts = "1726000000.000300"

        main.process("show tasks", user_id, channel_id, thread_ts)
        res = main.process("complete both", user_id, channel_id, thread_ts)
        assert "Which two tasks do you mean" in res


def test_thread_isolation():
    """Test that thread A and thread B have strictly isolated contexts."""
    items_store_a = [
        make_item("item_A1", "Thread A Task One", completed=False),
        make_item("item_A2", "Thread A Task Two", completed=False),
    ]
    items_store_b = [
        make_item("item_B1", "Thread B Task One", completed=False),
        make_item("item_B2", "Thread B Task Two", completed=False),
    ]

    all_items = items_store_a + items_store_b

    def mock_list_action_items(ctx, list_id):
        return [dict(x) for x in all_items]

    def mock_complete_action_item(item_id, ctx, list_id):
        for it in all_items:
            if it["id"] == item_id:
                for f in it["fields"]:
                    if f.get("column_id") == "col_completed":
                        f["checkbox"] = True
                        f["value"] = True
                return True
        return False

    with patch.object(slack_tools, "get_list_schema", return_value=SCHEMA), \
         patch.object(slack_tools, "list_action_items", side_effect=mock_list_action_items), \
         patch.object(slack_tools, "complete_action_item", side_effect=mock_complete_action_item), \
         patch.object(config, "has_permission", return_value=True):

        channel_id = "C_ISOLATION"
        user_id = "U12345"

        # Thread 1 shows tasks
        main.process("show tasks", user_id, channel_id, thread_ts="thread_111")
        
        # In Thread 2, user attempts 'complete the last' WITHOUT showing tasks in Thread 2
        res_t2 = main.process("complete the last", user_id, channel_id, thread_ts="thread_222")
        # Should raise / return error about no displayed tasks in this thread
        assert "couldn't find any recently displayed action items in this thread" in res_t2

        # Thread 1 'complete the last' succeeds for Thread 1
        res_t1 = main.process("complete the last", user_id, channel_id, thread_ts="thread_111")
        assert "Thread B Task Two" in res_t1  # (last item in all_items displayed in Thread 1)

