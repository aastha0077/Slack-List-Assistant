"""End-to-end offline tests through the real parser, resolver and Slack adapter."""
from copy import deepcopy
from datetime import datetime, timedelta
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import commands
import config
import delivery
import intent_parser
import main
import mutations
import progress_engine
import slack_tools
from references import parse_reference, select_ids, extract_contextual_reference
from references import TargetType


SCHEMA = {"schema": [
    {"id": "name", "key": "name", "type": "text"},
    {"id": "done", "key": "todo_completed", "type": "checkbox"},
    {"id": "owner", "key": "todo_assignee", "type": "user"},
    {"id": "due", "key": "todo_due_date", "type": "date"},
    {"id": "priority", "key": "priority", "type": "select", "options": {"choices": [
        {"id": "priority_" + str(i), "label": "P" + str(i)} for i in range(1, 5)]}},
]}


class FakeSlack:
    def __init__(self):
        self.items = []
        self.writes = []
        self.noop = False
        self.fail_field = None
        self.fail_create_name = None
        self.fail_read = False
        self.after_write = False
        self.next_id = 1
        self.list_requests = []
        self.users = [
            {"id": "UA", "name": "Alex"}, {"id": "UM", "name": "Morgan"},
            {"id": "UAA", "name": "Aastha"}, {"id": "UP", "name": "Praveen"}]

    def add(self, name, completed=False, assignee=None, priority="P2", due=None):
        item = {"id": f"I{self.next_id}", "fields": [{"column_id": "name", "text": name},
                {"column_id": "done", "checkbox": completed},
                {"column_id": "priority", "select": ["priority_" + priority[-1]]}]}
        if assignee:
            item["fields"].append({"column_id": "owner", "user": [assignee]})
        if due:
            item["fields"].append({"column_id": "due", "date": [due]})
        self.next_id += 1
        self.items.append(item)
        return item

    def slackLists_items_list(self, **kwargs):
        self.list_requests.append(kwargs.get("list_id"))
        if self.fail_read:
            raise RuntimeError("simulated read failure")
        return {"ok": True, "items": deepcopy(self.items), "list": {"list_metadata": SCHEMA}}

    def users_list(self, **kwargs):
        return {"ok": True, "members": deepcopy(self.users)}

    def users_info(self, user):
        found = next((entry for entry in self.users if entry["id"] == user), {"id": user, "name": user})
        return {"ok": True, "user": deepcopy(found)}

    def api_call(self, api_method, http_verb, json):
        self.writes.append((api_method, deepcopy(json)))
        if api_method.endswith("create"):
            name = slack_tools._rich_text(json["initial_fields"][0]["rich_text"])
            if name == self.fail_create_name:
                raise RuntimeError("simulated rejected create")
            item = {"id": f"I{self.next_id}", "fields": deepcopy(json["initial_fields"]) + [{"column_id": "done", "checkbox": False}]}
            self.next_id += 1
            if not self.noop:
                self.items.append(item)
            if self.after_write:
                raise TimeoutError()
            return {"ok": True, "item": deepcopy(item)}
        if api_method.endswith("delete"):
            if not self.noop:
                self.items[:] = [x for x in self.items if x["id"] != json["id"]]
        else:
            for cell in json["cells"]:
                if cell["column_id"] == self.fail_field:
                    raise RuntimeError("simulated rejected field")
                if not self.noop:
                    item = next(x for x in self.items if x["id"] == cell["row_id"])
                    item["fields"][:] = [f for f in item["fields"] if f["column_id"] != cell["column_id"]]
                    item["fields"].append(deepcopy(cell))
        if self.after_write:
            raise TimeoutError()
        return {"ok": True}


@pytest.fixture
def slack(monkeypatch):
    fake = FakeSlack()
    monkeypatch.setattr(slack_tools, "_client", fake)
    monkeypatch.setattr(config, "USER_ROLES", {"UA": "admin", "UM": "member", "UV": "viewer"})
    monkeypatch.setattr(config, "DEFAULT_ROLE", "viewer")
    monkeypatch.setattr(config, "CHANNEL_LISTS", {"C": "L", "C2": "L"})
    return fake


def ask(text, user="UA", channel="C", thread="T", msg=None, team="W"):
    return main.process(text, user, channel, thread, msg, team)


@pytest.mark.parametrize("reference,index", [
    ("first", 0), ("second", 1), ("third", 2), ("fourth", 3), ("fifth", 4), ("sixth", 5),
    ("1st", 0), ("2nd", 1), ("3rd", 2), ("4th", 3), ("5th", 4), ("6th", 5),
    ("first one", 0), ("second one", 1), ("last", 5), ("last one", 5), ("previous task", 5),
    ("task 6", 5), ("item 6", 5), ("number 6", 5), ("sixth task", 5),
    ("the second one from above", 1),
])
def test_immutable_ordinals(slack, reference, index):
    original = [slack.add(f"Work {i}") for i in range(6)]
    ask("show tasks")
    slack.items.reverse()
    response = ask("Can you complete " + reference + "?")
    assert "Action item completed" in response
    assert slack.writes[-1][1]["cells"][0]["row_id"] == original[index]["id"]


def test_deleted_displayed_position_does_not_shift(slack):
    original = [slack.add(f"Work {i}") for i in range(6)]
    ask("show tasks")
    slack.items.remove(original[0])
    assert "Work 5" in ask("What is the status of the 6th one?")
    assert "Action item completed" in ask("complete 6th")
    count = len(slack.writes)
    assert "no longer exists" in ask("complete first")
    assert len(slack.writes) == count


@pytest.mark.parametrize("reference", ["0th", "7th", "number 100", "first 8", "1 and 8"])
def test_invalid_positions_never_partially_mutate(slack, reference):
    slack.add("Alpha")
    ask("show tasks")
    assert "outside" in ask("complete " + reference)
    assert not slack.writes


@pytest.mark.parametrize("reference", ["it", "that task", "this task", "that", "this", "the task you just showed"])
def test_pronouns_after_creation(slack, reference):
    assert "created successfully" in ask("Create a task called Client Report with priority P2.")
    item_id = slack.items[0]["id"]
    assert "updated" in ask(f"Actually change {reference} to P1.")
    assert slack.writes[-1][1]["cells"][0]["row_id"] == item_id
    assert slack_tools.extract_priority(slack.items[0], SCHEMA) == "P1"


def test_focus_after_inspect_does_not_renumber(slack):
    items = [slack.add(f"Work {i}") for i in range(6)]
    ask("show tasks")
    assert "Work 5" in ask("What is the status of number 6?")
    ask("Change its priority to P1")
    assert slack.writes[-1][1]["cells"][0]["row_id"] == items[5]["id"]
    ask("complete first")
    assert slack.writes[-1][1]["cells"][0]["row_id"] == items[0]["id"]


def test_ambiguous_pronoun_does_not_guess(slack):
    slack.add("Alpha")
    slack.add("Beta")
    ask("show tasks")
    assert "Which task" in ask("complete it")
    assert not slack.writes


@pytest.mark.parametrize("phrase,count", [("both", 2), ("all", 3), ("those tasks", 3), ("those two", 2), ("first and third", 3)])
def test_displayed_sets(slack, phrase, count):
    items = [slack.add(f"Work {i}") for i in range(count)]
    ask("show tasks")
    response = ask("complete " + phrase)
    assert "completed" in response
    written = [payload["cells"][0]["row_id"] for _, payload in slack.writes]
    expected = [items[0]["id"], items[2]["id"]] if phrase == "first and third" else [x["id"] for x in items]
    assert written == expected


@pytest.mark.parametrize("title", ["First Review", "Last Client Report", "Second Phase Review", "All Hands Meeting", "Ship it now", "This quarter planning"])
def test_title_words_are_not_references(slack, title):
    wrong = slack.add("Unrelated")
    target = slack.add(title)
    ask("show tasks")
    response = ask("complete " + title)
    assert "Action item completed" in response
    assert slack.writes[-1][1]["cells"][0]["row_id"] == target["id"]
    assert not slack_tools.extract_completed(wrong, SCHEMA)


def test_quoted_reference_word_is_literal_title(slack):
    target = slack.add("First")
    assert "completed" in ask('complete "First"')
    assert slack.writes[-1][1]["cells"][0]["row_id"] == target["id"]


@pytest.mark.parametrize("reply", ["second", "the second one", "2nd", "number 2"])
def test_clarification_preserves_duplicate_name_id(slack, reply):
    first, second = slack.add("Report"), slack.add("Report")
    assert "Which Report" in ask("complete Report")
    # Rename selected candidate after clarification; the original ID still wins.
    second["fields"][0]["text"] = "Renamed after question"
    response = ask(reply)
    assert "completed" in response
    assert slack.writes[-1][1]["cells"][0]["row_id"] == second["id"]
    assert not slack_tools.extract_completed(first, SCHEMA)


@pytest.mark.parametrize("reply", ["both", "all"])
def test_clarification_sets_without_previous_view(slack, reply):
    slack.add("Report alpha")
    slack.add("Report beta")
    slack.add("Unrelated")
    assert "Which report" in ask("complete report")
    assert "completed" in ask(reply)
    assert len(slack.writes) == 2


@pytest.mark.parametrize("changes", [dict(thread="OTHER"), dict(user="UM"), dict(channel="C2"), dict(team="W2")])
def test_scope_isolation(slack, changes):
    slack.add("Alpha")
    ask("show tasks", thread=None)
    ask("show tasks")
    assert "recently displayed" in ask("complete first", **changes)
    assert not slack.writes


def test_list_isolation(slack, monkeypatch):
    slack.add("Alpha")
    ask("show tasks")
    monkeypatch.setitem(config.CHANNEL_LISTS, "C", "DIFFERENT_LIST")
    assert "recently displayed" in ask("complete first")
    assert not slack.writes


@pytest.mark.parametrize("root", ["USER_ROOT", "BOT_ROOT"])
def test_reply_to_user_or_bot_root_and_restart(slack, root):
    slack.add("Alpha")
    ask("show tasks", thread=None, msg="USER_ROOT")
    main.record_bot_response("C", "BOT_ROOT", msg_ts="USER_ROOT", user_id="UA", team_id="W")
    main._pending.clear()
    assert "completed" in ask("complete first", thread=root, msg="REPLY")


def test_older_bot_response_retains_its_snapshot(slack):
    first = slack.add("Alpha")
    ask("show tasks", thread=None, msg="M1")
    main.record_bot_response("C", "B1", msg_ts="M1", user_id="UA", team_id="W")
    slack.add("Beta")
    ask("show tasks", thread=None, msg="M2")
    main.record_bot_response("C", "B2", msg_ts="M2", user_id="UA", team_id="W")
    ask("complete last", thread="B1", msg="M3")
    assert slack.writes[-1][1]["cells"][0]["row_id"] == first["id"]


def test_context_expiry(slack, monkeypatch):
    slack.add("Alpha")
    ask("show tasks")
    now = main.time.time()
    monkeypatch.setattr(main.time, "time", lambda: now + main._CONTEXT_TTL + 1)
    assert "recently displayed" in ask("complete first")
    assert not slack.writes


@pytest.mark.parametrize("phrase", ["Who is assigned to this?", "When is this due?", "What is the priority of that task?", "Did I finish Client Report?"])
def test_information_questions_do_not_mutate(slack, phrase):
    slack.add("Client Report", assignee="UA")
    ask("show tasks")
    assert "Client Report" in ask(phrase)
    assert not slack.writes


@pytest.mark.parametrize("phrase", ["What do I still have to finish?", "What are my pending tasks?", "show my tasks", "what do I still have?"])
def test_self_pending_queries(slack, phrase):
    slack.add("Pending mine", assignee="UA")
    slack.add("Completed mine", assignee="UA", completed=True)
    slack.add("Other user", assignee="UM")
    response = ask(phrase)
    assert "Pending mine" in response
    assert "Completed mine" not in response
    assert "Other user" not in response


def test_all_open_completed_and_count(slack):
    slack.add("Alpha")
    slack.add("Beta", completed=True)
    assert "Beta" in ask("show all")
    assert "Beta" not in ask("Give me all open items")
    assert "Beta" in ask("show completed tasks")
    assert "Alpha" not in ask("show completed tasks")
    assert "1 matching" in ask("How many tasks are open?")


@pytest.mark.parametrize("phrase", [
    "show every completed and pending task",
    "display both pending and finished items",
    "list all open and done work",
])
def test_multiple_status_list_queries_preserve_every_requested_state(slack, phrase):
    slack.add("Open record")
    slack.add("Closed record", completed=True)
    response = ask(phrase)
    assert "Open record" in response
    assert "Closed record" in response


@pytest.mark.parametrize("phrase,shown,hidden", [
    ("display unfinished work", "Open record", "Closed record"),
    ("list finished items", "Closed record", "Open record"),
])
def test_single_status_queries_remain_single_filters(slack, phrase, shown, hidden):
    slack.add("Open record")
    slack.add("Closed record", completed=True)
    response = ask(phrase)
    assert shown in response
    assert hidden not in response


@pytest.mark.parametrize("phrase", [
    "compare open versus finished tasks",
    "give me completed and pending counts",
    "show the difference between done and unfinished work",
])
def test_status_comparison_uses_complete_current_list_snapshot(slack, phrase):
    slack.add("Open one")
    slack.add("Open two")
    slack.add("Closed one", completed=True)
    response = ask(phrase)
    assert "Status distribution" in response
    assert "Pending:" in response and " 2" in response
    assert "Completed:" in response and " 1" in response


def test_empty_status_comparison_renders_zero_counts(slack):
    response = ask("compare finished versus unfinished items")
    assert "Pending:" in response and " 0" in response
    assert "Completed:" in response and " 0" in response


@pytest.mark.parametrize("phrase,expected_due", [
    ("create release checks due 2026-09-30", "2026-09-30"),
    ("add release checks to date 2026-09-30", "2026-09-30"),
    ("create release checks for September 30", "2026-09-30"),
    ("add release checks due next Friday", "2026-09-25"),
])
def test_creation_date_clause_is_a_field_not_title_or_assignee(slack, phrase, expected_due):
    response = ask(phrase)
    assert "created successfully" in response
    assert slack_tools.extract_item_name(slack.items[-1], SCHEMA) == "release checks"
    assert slack_tools.extract_due_date(slack.items[-1], SCHEMA) == expected_due
    assert slack_tools.extract_assignee_ids(slack.items[-1], SCHEMA) == []


@pytest.mark.parametrize("phrase", ["What's the weather?", "Tell me a joke", "Who is the president?", "Show my horoscope", "Explain Python decorators", "What is the capital of France?"])
def test_unrelated_scope(slack, phrase):
    slack.add("Alpha")
    response = ask(phrase)
    assert "only help with Slack List" in response
    assert not slack.writes


def test_all_mutations_use_real_adapter_and_verification(slack):
    assert "created successfully" in ask("create Client Report priority P2")
    assert "updated" in ask("change it to P1")
    assert "completed" in ask("Mark that task as done")
    assert "reopened" in ask("Reopen the previous task")
    assert "deleted" in ask("delete it")
    assert not slack.items
    assert "no longer exists" in ask("complete it")


@pytest.mark.parametrize("command", ["complete Alpha", "reopen Alpha", "delete Alpha", "change Alpha to P1"])
def test_noop_write_is_not_success(slack, command):
    slack.add("Alpha", completed=command.startswith("reopen"))
    slack.noop = True
    result = ask(command)
    assert "not all changes verified" in result


def test_unverified_create(slack):
    slack.noop = True
    assert "Creation not verified" in ask("create Alpha")


def test_partial_field_failure(slack):
    item = slack.add("Alpha")
    slack.fail_field = "priority"
    ctx = config.build_context("UA", "C", team_id="W", thread_ts="T")
    result = main.handle_mutation({"intent": "update", "task_name": "Alpha", "changes": [
        {"field": "name", "value": "Renamed"}, {"field": "priority", "value": "P1"}]}, ctx, "unused")
    assert "not all changes verified" in result
    assert "some fields may have changed" in result
    assert slack_tools.extract_item_name(item, SCHEMA) == "Renamed"


def test_create_multiple_individual_metadata_and_partial_failure(slack):
    slack.fail_create_name = "Beta"
    result = ask("Create tasks:\n- Alpha\n- Beta\n- Gamma")
    assert "Alpha" in result and "Gamma" in result and "Beta" in result
    assert "could not be created" in result
    assert [slack_tools.extract_item_name(x, SCHEMA) for x in slack.items] == ["Alpha", "Gamma"]


def test_multi_completion(slack):
    slack.add("Alpha")
    slack.add("Beta")
    result = ask("I finished Alpha and completed Beta")
    assert "Alpha" in result and "Beta" in result
    assert all(slack_tools.extract_completed(x, SCHEMA) for x in slack.items)


def test_duplicate_create_does_not_reassign(slack):
    ask("create Alpha")
    before = len(slack.writes)
    assert "already exists" in ask("create Alpha")
    assert len(slack.writes) == before
    assert "different fields" in ask("create Alpha priority P1")
    assert len(slack.writes) == before
    ask("create Alpha for Morgan")
    assert len(slack.items) == 2
    assert slack_tools.extract_assignee_id(slack.items[0], SCHEMA) is None


def test_similar_names_not_duplicates(slack):
    ask("create Client Report")
    ask("create Client Report Review")
    assert len(slack.items) == 2


@pytest.mark.parametrize("template", ["create Alpha due {}", "set deadline of Alpha to {}"])
def test_past_dates_no_writes(slack, template):
    slack.add("Alpha")
    past = (main.current_date() - timedelta(days=1)).isoformat()
    assert "past" in ask(template.format(past))
    assert not slack.writes


@pytest.mark.parametrize("phrase", ["today", "tomorrow", "Friday", "next Monday", "next week"])
def test_natural_dates(slack, phrase):
    result = ask("create Alpha due " + phrase)
    assert "created successfully" in result
    assert slack_tools.extract_due_date(slack.items[0], SCHEMA) >= main.current_date().isoformat()


@pytest.mark.parametrize("user,command", [("UV", "create Alpha"), ("UV", "complete Alpha"),
    ("UV", "change Alpha to P1"), ("UM", "delete Alpha"), ("UM", "change Alpha to P1"),
    ("UM", "reassign Alpha to Alex"), ("UM", "create Beta for Alex")])
def test_permissions_no_writes(slack, user, command):
    slack.add("Alpha")
    assert "Permission denied" in ask(command, user=user)
    assert not slack.writes


def test_permission_change_is_dynamic(slack, monkeypatch):
    slack.add("Alpha")
    assert "Permission denied" in ask("delete Alpha", user="UM")
    monkeypatch.setitem(config.USER_ROLES, "UM", "admin")
    assert "deleted" in ask("delete Alpha", user="UM")


def test_field_preflight_prevents_partial_unauthorized_write(slack):
    slack.add("Alpha")
    ctx = config.build_context("UM", "C")
    with pytest.raises(PermissionError):
        main.handle_mutation({"intent": "update", "task_name": "Alpha", "changes": [
            {"field": "name", "value": "Renamed"}, {"field": "priority", "value": "P1"}]}, ctx, "x")
    assert not slack.writes


def test_pagination(monkeypatch):
    client = Mock()
    client.slackLists_items_list.side_effect = [
        {"ok": True, "items": [{"id": "A"}], "response_metadata": {"next_cursor": "NEXT"}},
        {"ok": True, "items": [{"id": "B"}]}]
    monkeypatch.setattr(slack_tools, "_client", client)
    assert [x["id"] for x in slack_tools.list_action_items(list_id="L")] == ["A", "B"]
    assert client.slackLists_items_list.call_args.kwargs["cursor"] == "NEXT"


def test_schema_identity_precedes_generic_type():
    schema = {"schema": [{"id": "other", "type": "select", "key": "status"}] + SCHEMA["schema"]}
    cell = slack_tools._write_cell(schema, "priority", "P1")
    assert cell["column_id"] == "priority"
    assert slack_tools.column({"schema": [{"id": "other", "type": "select", "key": "priority"}]}, keys={"status"}, types={"select"}) is None


def test_failed_verification_read_is_unknown(slack):
    item = slack.add("Alpha")
    slack.fail_read = True
    result = mutations.verify(item["id"], [{"field": "completed", "value": True}], config.build_context("UA", "C"), SCHEMA)
    assert not result.verified
    assert "unknown" in result.problems[0]


def test_retry_same_event_never_repeats_mutation(slack):
    send = Mock(return_value={"ts": "BOT"})
    for _ in range(2):
        delivery.execute_event(main._db, "event-1", lambda: ask("create Alpha"), send)
    assert len(slack.items) == 1
    assert len(slack.writes) == 1
    assert send.call_count == 1


def test_retry_failed_post_reuses_response(slack):
    send = Mock(side_effect=[TimeoutError(), {"ts": "BOT"}])
    run = Mock(side_effect=lambda: ask("create Alpha"))
    with pytest.raises(TimeoutError):
        delivery.execute_event(main._db, "event-2", run, send)
    delivery.execute_event(main._db, "event-2", run, send, lambda: None)
    assert run.call_count == 1
    assert len(slack.writes) == 1


def test_retry_post_that_already_arrived_does_not_duplicate(slack):
    send = Mock(side_effect=TimeoutError())
    with pytest.raises(TimeoutError):
        delivery.execute_event(main._db, "event-3", lambda: "answer", send)
    delivery.execute_event(main._db, "event-3", lambda: pytest.fail("must not rerun"), send, lambda: "EXISTING_BOT")
    assert send.call_count == 1


def test_retry_read_failure_can_recover(slack):
    send = Mock(return_value={"ts": "BOT"})
    slack.fail_read = True
    with pytest.raises(RuntimeError):
        delivery.execute_event(main._db, "event-4", lambda: ask("create Alpha"), send)
    slack.fail_read = False
    delivery.execute_event(main._db, "event-4", lambda: ask("create Alpha"), send)
    assert len(slack.items) == 1


def test_create_timeout_after_commit_reconciles(slack):
    slack.after_write = True
    response = ask("create Alpha")
    assert "created successfully" in response
    assert len(slack.items) == 1


def test_untrusted_ids_are_removed():
    parsed = commands.validate_command({"intent": "complete", "task_name": "Alpha", "target_ids": ["WRONG"]})
    assert "target_ids" not in parsed


def test_user_mentions_are_not_bot_mentions(slack, monkeypatch):
    deliver = Mock()
    monkeypatch.setattr(main, "_deliver", deliver)
    event = {"user": "UA", "channel": "C", "thread_ts": "T", "ts": "M", "text": "reassign Alpha to <@UM>"}
    main.message_handler({"team_id": "W"}, event, {"bot_user_id": "UBOT"})
    assert deliver.call_count == 1
    assert "<@UM>" in deliver.call_args.args[1]


def test_slash_add_preserves_creation_intent(slack, monkeypatch):
    deliver = Mock()
    monkeypatch.setattr(main, "_deliver", deliver)
    ack = Mock()
    main.add_command(ack, {"user_id": "UA", "channel_id": "C", "team_id": "W", "trigger_id": "TR", "text": "Client Report"})
    ack.assert_called_once()
    assert deliver.call_args.args[1] == "add Client Report"


def test_high_ordinals_general_grammar():
    assert select_ids(parse_reference("twenty-third task"), list(range(30))) == [22]
    assert select_ids(parse_reference("101st"), list(range(110))) == [100]
    assert select_ids(parse_reference("first two"), list(range(5))) == [0, 1]


def test_mixed_contextual_positions_preserve_exact_order():
    displayed = ["I1", "I2", "I3", "I4"]
    assert select_ids(parse_reference("first, third and last"), displayed) == ["I1", "I3", "I4"]
    assert select_ids(parse_reference("2nd and final"), displayed) == ["I2", "I4"]


def test_relative_next_uses_exact_focused_id_and_requires_focus():
    displayed = ["I1", "I2", "I3"]
    assert select_ids(parse_reference("next one"), displayed, ["I2"]) == ["I3"]
    with pytest.raises(ValueError, match="starting point"):
        select_ids(parse_reference("next"), displayed)


def test_both_and_demonstratives_prefer_recent_selected_set():
    displayed = ["I1", "I2", "I3", "I4"]
    focus = ["I1", "I3"]
    assert select_ids(parse_reference("both"), displayed, focus) == focus
    assert select_ids(parse_reference("those tasks"), displayed, focus) == focus


def test_natural_next_reference_mutates_adjacent_displayed_item(slack):
    items = [slack.add("Alpha"), slack.add("Beta"), slack.add("Gamma")]
    ask("show tasks")
    assert "Beta" in ask("What is the status of the second one?")
    response = ask("complete the next one")
    assert "completed" in response
    assert slack.writes[-1][1]["cells"][0]["row_id"] == items[2]["id"]


@pytest.mark.parametrize("reference,index", [
    ("first", 0), ("second", 1), ("2nd", 1),
    ("third", 2), ("3rd", 2), ("last", 2),
])
def test_member_qualified_ordinal_resolves_within_filtered_order(slack, reference, index):
    praveen = [slack.add("gamma2", assignee="UP"),
               slack.add("gamma1", assignee="UP"),
               slack.add("gamma", assignee="UP")]
    unrelated = slack.add("unrelated", assignee="UM")
    slack.items.remove(unrelated)
    slack.items.insert(1, unrelated)
    response = ask(f"what is the status of @Praveen {reference} task")
    assert slack_tools.extract_item_name(praveen[index], SCHEMA) in response
    assert "unrelated" not in response


def test_member_qualified_ordinal_mutation_passes_exact_filtered_id(slack):
    first = slack.add("Praveen one", assignee="UP")
    slack.add("Other member", assignee="UM")
    second = slack.add("Praveen two", assignee="UP")
    response = ask("complete @Praveen second task")
    assert "completed" in response
    assert slack.writes[-1][1]["cells"][0]["row_id"] == second["id"]
    assert not slack_tools.extract_completed(first, SCHEMA)


def test_embedded_reference_normalization_preserves_filters_and_targeted_intent():
    parsed = commands.validate_command(intent_parser.parse_intent(
        "show me the second task assigned to @Praveen"))
    assert parsed["intent"] == "inspect"
    assert parsed["assignees"] == ["@Praveen"]
    assert parsed["reference"] == {"kind": "positions", "positions": (2,), "count": 0}
    assert parsed["reference_scope"] == "filtered"


def test_embedded_multiple_positions_are_one_exact_reference_set():
    reference = extract_contextual_reference("please use the first and third displayed tasks")
    assert select_ids(reference, ["I1", "I2", "I3", "I4"]) == ["I1", "I3"]


def test_reference_extractor_does_not_reinterpret_task_title_words():
    assert extract_contextual_reference("complete First Review") is None
    assert extract_contextual_reference("inspect Last Client Report") is None
    assert extract_contextual_reference("show the status of gamma1 task") is None


@pytest.mark.parametrize("task_name", ["gamma", "gamma1"])
def test_member_filtered_named_status_is_targeted_not_list(slack, task_name):
    slack.add("gamma2", assignee="UP")
    target = slack.add(task_name, assignee="UP")
    slack.add("Other member task", assignee="UM")
    response = ask(f"show me status of @Praveen {task_name} task")
    assert task_name in response
    assert "gamma2" not in response
    state = main._state(main.context("UA", "C", "T", None, "W"))
    assert state["focus_ids"] == [target["id"]]


@pytest.mark.parametrize("task_name", ["gamma", "gamma1"])
def test_slack_mrkdwn_member_target_remains_single_item(slack, task_name):
    slack.add("gamma2", assignee="UP")
    target = slack.add(task_name, assignee="UP")
    parsed = commands.validate_command(intent_parser.parse_intent(
        f"show me status of <@UP> *{task_name} task*"))
    assert parsed["intent"] == "inspect"
    assert parsed["task_name"] == task_name
    response = ask(f"show me status of <@UP> *{task_name} task*")
    assert task_name in response and "gamma2" not in response
    state = main._state(main.context("UA", "C", "T", None, "W"))
    assert state["focus_ids"] == [target["id"]]


def test_member_filtered_last_status_selects_last_exact_id(slack):
    tasks = [slack.add("gamma2", assignee="UP"), slack.add("gamma1", assignee="UP"),
             slack.add("gamma", assignee="UP")]
    slack.add("Other member", assignee="UM")
    ask("show all tasks assigned to @Praveen")
    response = ask("whats the status of @Praveen last task")
    assert "gamma" in response and "gamma1" not in response and "gamma2" not in response
    state = main._state(main.context("UA", "C", "T", None, "W"))
    assert state["focus_ids"] == [tasks[-1]["id"]]


def test_member_filtered_ambiguous_name_asks_clarification(slack):
    slack.add("Weekly Report", assignee="UP")
    slack.add("Weekly Report", assignee="UP")
    response = ask("show me status of @Praveen Weekly Report task")
    assert "Which Weekly Report task" in response


def test_member_filtered_missing_name_returns_not_found(slack):
    slack.add("Existing", assignee="UP")
    response = ask("show me status of @Praveen Missing task")
    assert "couldn't find that action item" in response


@pytest.mark.parametrize("phrase", ["finish everything assigned to me", "mark all my pending work as done"])
def test_broad_self_scoped_completion(slack, phrase):
    mine = [slack.add("Mine A", assignee="UA"), slack.add("Mine B", assignee="UA")]
    slack.add("Someone else's", assignee="UM")
    response = ask(phrase)
    assert "completed" in response
    assert all(slack_tools.extract_completed(item, SCHEMA) for item in mine)
    assert not slack_tools.extract_completed(slack.items[2], SCHEMA)


def test_broad_self_scoped_delete_respects_permissions(slack):
    slack.add("Mine", assignee="UM")
    assert "Permission denied" in ask("remove everything on my list", user="UM")
    assert not slack.writes


def test_assign_first_two(slack):
    items = [slack.add("Alpha"), slack.add("Beta"), slack.add("Gamma")]
    ask("show tasks")
    response = ask("assign the first two to @Morgan")
    assert "updated" in response
    assert [slack_tools.extract_assignee_id(item, SCHEMA) for item in items] == ["UM", "UM", None]


@pytest.mark.parametrize("phrase,reference_kind,count", [
    ("assign the first task to me", "positions", 1),
    ("assign the first two tasks to me", "head", 2),
    ("move Praveen's last task to me", "positions", 1),
    ("assign the second and fourth tasks to Aastha", "positions", 2),
    ("give both of those tasks to Praveen", "both", 2),
])
def test_assignment_selection_is_preserved_separately_from_destination(phrase, reference_kind, count):
    parsed = commands.validate_command(intent_parser.parse_intent(phrase))
    assert parsed["intent"] == "update"
    assert parsed["reference"]["kind"] == reference_kind
    assert parsed["changes"][0]["field"] == "assignee"
    assert parsed["target_selection"]["count"] == count


def test_filtered_bulk_assignment_resolves_exact_ids_without_display_context(slack, monkeypatch):
    wanted = [slack.add("P1 pending one", assignee="UP", priority="P1"),
              slack.add("P1 pending two", assignee="UP", priority="P1")]
    slack.add("Wrong priority", assignee="UP", priority="P2")
    slack.add("Already complete", assignee="UP", priority="P1", completed=True)
    slack.add("Wrong owner", assignee="UM", priority="P1")
    captured = []
    original = mutations.execute_collection

    def record(item_ids, intent, changes, ctx, schema):
        captured.extend(item_ids)
        return original(item_ids, intent, changes, ctx, schema)

    monkeypatch.setattr(mutations, "execute_collection", record)
    response = ask("Assign all pending P1 tasks belonging to Praveen to me", thread="FILTERED_ASSIGN")
    assert "Action items updated" in response
    assert captured == [item["id"] for item in wanted]
    assert all(slack_tools.extract_assignee_ids(item, SCHEMA) == ["UA"] for item in wanted)


def test_filtered_selection_assigns_only_first_two_exact_ids(slack):
    wanted = [slack.add(f"Praveen {index}", assignee="UP") for index in range(1, 4)]
    slack.add("Other owner", assignee="UM")
    response = ask("Move the first two of Praveen's pending tasks to me", thread="FILTERED_HEAD")
    assert "Action items updated" in response
    assert [slack_tools.extract_assignee_ids(item, SCHEMA) for item in wanted] == [["UA"], ["UA"], ["UP"]]


def test_filtered_last_assignment_is_singular_and_context_followup_keeps_id(slack):
    tasks = [slack.add(f"Praveen {index}", assignee="UP") for index in range(1, 4)]
    response = ask("Move Praveen's last task to me", thread="FILTERED_LAST")
    assert "Action item updated" in response
    assert [slack_tools.extract_assignee_ids(item, SCHEMA) for item in tasks] == [["UP"], ["UP"], ["UA"]]

    response = ask("Actually give it back to Praveen", thread="FILTERED_LAST")
    assert "Action item updated" in response
    assert slack_tools.extract_assignee_ids(tasks[-1], SCHEMA) == ["UP"]


def test_assignment_verification_rejects_unobserved_assignee_change(slack):
    item = slack.add("Alpha")
    ask("show tasks", thread="VERIFY_ASSIGN")
    slack.noop = True
    response = ask("assign the first task to me", thread="VERIFY_ASSIGN")
    assert "not all changes verified" in response
    assert slack_tools.extract_assignee_ids(item, SCHEMA) == []


@pytest.mark.parametrize("suffix,expected_fields", [
    ("", {"assignee"}),
    (" due Friday", {"assignee", "due_date"}),
    (" p3", {"assignee", "priority"}),
    (" due Friday p3", {"assignee", "due_date", "priority"}),
])
def test_named_assignment_metadata_is_structured_not_part_of_title(suffix, expected_fields):
    parsed = commands.validate_command(intent_parser.parse_intent(
        f"Assign the arbitrary verification task to @DynamicUser{suffix}"))
    assert parsed["intent"] == "update"
    assert parsed["task_name"] == "arbitrary verification task"
    assert {change["field"] for change in parsed["changes"]} == expected_fields
    assert parsed["changes"][0] == {"field": "assignee", "value": "@DynamicUser"}


@pytest.mark.parametrize("date_text", [
    "today", "tomorrow", "Friday", "next Monday", "September 25", "2026-09-25",
])
def test_assignment_natural_dates_are_normalized(date_text):
    parsed = commands.validate_command(intent_parser.parse_intent(
        f"Assign Release validation to Morgan due {date_text}"))
    due = next(change["value"] for change in parsed["changes"] if change["field"] == "due_date")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", due)


def test_named_assignment_updates_and_verifies_all_requested_fields(slack):
    slack.users.append({"id": "UDYNAMIC", "name": "DynamicUser"})
    item = slack.add("arbitrary verification task")
    response = ask("Assign the arbitrary verification task to @DynamicUser due Friday p3",
                   thread="ASSIGN_FIELDS")
    assert "Action item updated" in response
    assert slack_tools.extract_assignee_ids(item, SCHEMA) == ["UDYNAMIC"]
    assert slack_tools.extract_due_date(item, SCHEMA) == intent_parser._resolve_natural_date("Friday")
    assert slack_tools.extract_priority(item, SCHEMA) == "P3"


def test_compound_assignments_execute_independently_with_named_and_filtered_targets(slack):
    slack.users.append({"id": "UASTHAA", "name": "AasthaA"})
    named = slack.add("testing task")
    filtered = [slack.add("Praveen first", assignee="UP"),
                slack.add("Praveen second", assignee="UP")]
    text = ("Assign the testing task to @AasthaA due Friday p3 and "
            "Assign @Praveen first task to me")
    parsed = commands.validate_command(intent_parser.parse_intent(text))
    assert parsed["intent"] == "compound"
    assert len(parsed["operations"]) == 2
    response = ask(text, thread="COMPOUND_ASSIGN")
    assert response.count("Action item updated") == 2
    assert slack_tools.extract_assignee_ids(named, SCHEMA) == ["UASTHAA"]
    assert slack_tools.extract_due_date(named, SCHEMA) == intent_parser._resolve_natural_date("Friday")
    assert slack_tools.extract_priority(named, SCHEMA) == "P3"
    assert slack_tools.extract_assignee_ids(filtered[0], SCHEMA) == ["UA"]
    assert slack_tools.extract_assignee_ids(filtered[1], SCHEMA) == ["UP"]


def test_compound_assignment_reports_partial_failure_without_undoing_success(slack):
    slack.users.append({"id": "UASTHAA", "name": "AasthaA"})
    named = slack.add("testing task")
    filtered = slack.add("Only Praveen task", assignee="UP")
    response = ask(
        "Assign the testing task to @AasthaA and assign @Praveen fourth task to me",
        thread="COMPOUND_PARTIAL")
    assert "Action item updated" in response
    assert "outside" in response
    assert slack_tools.extract_assignee_ids(named, SCHEMA) == ["UASTHAA"]
    assert slack_tools.extract_assignee_ids(filtered, SCHEMA) == ["UP"]


@pytest.mark.parametrize("connector", ["additionally", "after that", "then", "and also"])
def test_independent_operation_connectors_produce_compound_intent(connector):
    parsed = commands.validate_command(intent_parser.parse_intent(
        f"assign Alpha to Morgan {connector} complete Beta"))
    assert parsed["intent"] == "compound"
    assert [operation["intent"] for operation in parsed["operations"]] == ["update", "complete"]


@pytest.mark.parametrize("priority_text,expected", [
    ("urgent", "P1"), ("high priority", "P1"), ("medium", "P2"), ("low priority", "P4"),
])
def test_assignment_priority_continuations_normalize_semantically(priority_text, expected):
    parsed = commands.validate_command(intent_parser.parse_intent(
        f"give Morgan the Release readiness task and make it {priority_text}"))
    priority = next(change["value"] for change in parsed["changes"] if change["field"] == "priority")
    assert priority == expected


def test_another_is_relative_to_exact_focused_item(slack):
    items = [slack.add(f"Work {index}") for index in range(3)]
    ask("show tasks", thread="ANOTHER_REFERENCE")
    ask("what is the status of the first task", thread="ANOTHER_REFERENCE")
    response = ask("complete another task", thread="ANOTHER_REFERENCE")
    assert "Action item completed" in response
    assert slack_tools.extract_completed(items[1], SCHEMA)
    assert not slack_tools.extract_completed(items[0], SCHEMA)
    assert not slack_tools.extract_completed(items[2], SCHEMA)


@pytest.mark.parametrize("assignee_phrase", [
    "someone else", "another person", "someone other than me", "someone else on the team",
])
def test_relative_assignee_language_normalizes_to_other_condition(assignee_phrase):
    parsed = commands.validate_command(intent_parser.parse_intent(
        f"show open P1 tasks assigned to {assignee_phrase}"))
    assert parsed["intent"] == "list"
    assert parsed["assignee_condition"] == "other"
    assert not parsed.get("assignees")
    assert parsed["priority"] == "P1"
    assert parsed["completed"] is False


@pytest.mark.parametrize("wording,expected", [
    ("show high priority tasks", "P1"),
    ("list urgent work", "P1"),
    ("find medium priority items", "P2"),
    ("show low priority tasks", "P4"),
])
def test_qualitative_priority_filters_are_normalized_for_general_queries(wording, expected):
    parsed = commands.validate_command(intent_parser.parse_intent(wording))
    assert parsed["priority"] == expected


@pytest.mark.parametrize("phrase,condition", [
    ("show unassigned tasks", "unassigned"),
    ("list tasks with no owner", "unassigned"),
    ("find tasks with any assignee", "assigned"),
    ("show work assigned to anyone", "assigned"),
])
def test_non_specific_assignee_conditions_are_not_member_names(phrase, condition):
    parsed = commands.validate_command(intent_parser.parse_intent(phrase))
    assert parsed["assignee_condition"] == condition
    assert not parsed.get("assignees")


def test_relative_assignee_filter_composes_with_priority_status_and_due(slack):
    due = main.current_date().isoformat()
    wanted = slack.add("Other urgent today", assignee="UM", priority="P1", due=due)
    slack.add("Mine urgent today", assignee="UA", priority="P1", due=due)
    slack.add("Unassigned urgent today", priority="P1", due=due)
    slack.add("Other lower priority", assignee="UM", priority="P2", due=due)
    slack.add("Other completed", assignee="UM", priority="P1", due=due, completed=True)
    response = ask("show open P1 tasks due today assigned to another person", thread="OTHER_FILTER")
    assert "Other urgent today" in response
    for excluded in ("Mine urgent today", "Unassigned urgent today", "Other lower priority", "Other completed"):
        assert excluded not in response
    state = main._state(main.context("UA", "C", "OTHER_FILTER", None, "W"))
    assert [item["id"] for item in state["items"]] == [wanted["id"]]


def test_member_cannot_use_relative_assignee_filter_to_view_others(slack):
    slack.add("Another member's work", assignee="UA", priority="P1")
    response = ask("show P1 tasks assigned to someone else", user="UM", thread="MEMBER_OTHER")
    assert "Permission denied" in response


def test_assignment_structure_keeps_arbitrary_target_and_fields_separate():
    parsed = commands.validate_command(intent_parser.parse_intent(
        "transfer Regional launch readiness to @DynamicOwner due next Monday low priority"))
    assert parsed["task_name"] == "Regional launch readiness"
    assert parsed["changes"][0] == {"field": "assignee", "value": "@DynamicOwner"}
    assert {change["field"] for change in parsed["changes"]} == {"assignee", "due_date", "priority"}


def test_quantity_prefixed_multi_create_executes_every_structured_task(slack):
    slack.users.append({"id": "UOWNER", "name": "OwnerX"})
    response = ask(
        "Add two tasks: inspect telemetry and validate gateway to @OwnerX due 2026-10-11 p2",
        thread="QUANTITY_MULTI_CREATE")
    assert "could not be created" not in response
    assert [slack_tools.extract_item_name(item, SCHEMA) for item in slack.items] == [
        "inspect telemetry", "validate gateway"]
    assert all(slack_tools.extract_assignee_ids(item, SCHEMA) == ["UOWNER"] for item in slack.items)
    assert all(slack_tools.extract_due_date(item, SCHEMA) == "2026-10-11" for item in slack.items)
    assert all(slack_tools.extract_priority(item, SCHEMA) == "P2" for item in slack.items)


def test_contextual_ordinal_filters_immutable_display_before_selection(slack):
    slack.users.append({"id": "UOWNER", "name": "OwnerX"})
    owned = [slack.add(f"Owned {index}", assignee="UOWNER") for index in range(3)]
    slack.add("Different owner", assignee="UM")
    ask("show all tasks", thread="FILTERED_DISPLAY")
    slack.items.reverse()
    response = ask(
        "Assign the second one of @OwnerX to Morgan",
        thread="FILTERED_DISPLAY")
    assert "Action item updated" in response
    assert slack_tools.extract_assignee_ids(owned[1], SCHEMA) == ["UM"]
    assert slack_tools.extract_assignee_ids(owned[0], SCHEMA) == ["UOWNER"]
    assert slack_tools.extract_assignee_ids(owned[2], SCHEMA) == ["UOWNER"]


def test_contextual_filtered_completion_reports_refetched_completed_state(slack):
    slack.users.append({"id": "UOWNER", "name": "OwnerX"})
    selected = slack.add("Selected work", assignee="UOWNER")
    slack.add("Other work", assignee="UOWNER")
    ask("show all tasks", thread="FILTERED_COMPLETE")
    response = ask("Mark the first one as done of @OwnerX", thread="FILTERED_COMPLETE")
    assert "Action item completed" in response
    assert "Status: Completed" in response
    assert "Status: Pending" not in response
    assert slack_tools.extract_completed(selected, SCHEMA)


def test_response_fails_closed_on_inconsistent_verified_completion(slack, monkeypatch):
    item = slack.add("Guarded completion", assignee="UA")
    ask("show tasks", thread="INCONSISTENT_VERIFY")

    def inconsistent(item_ids, intent, changes, ctx, schema):
        return [mutations.MutationResult(
            item_id=item_ids[0], item=deepcopy(item), verified=True,
            outcome="verified_success")]

    monkeypatch.setattr(mutations, "execute_collection", inconsistent)
    response = ask("complete the first one", thread="INCONSISTENT_VERIFY")
    assert "not all changes verified" in response
    assert "still shows pending" in response
    assert "Action item completed" not in response


def test_named_assignment_resolves_safe_normalized_match_to_exact_id(slack, monkeypatch):
    item = slack.add("validate service gateway", priority="P1", due="2026-10-12")
    captured = []
    original = mutations.execute_collection

    def record(item_ids, intent, changes, ctx, schema):
        captured.extend(item_ids)
        return original(item_ids, intent, changes, ctx, schema)

    monkeypatch.setattr(mutations, "execute_collection", record)
    response = ask(
        "Assign the service gateway validation task to Morgan",
        thread="NAMED_NORMALIZED")
    assert "Action item updated" in response
    assert captured == [item["id"]]
    assert slack_tools.extract_assignee_ids(item, SCHEMA) == ["UM"]
    assert slack_tools.extract_priority(item, SCHEMA) == "P1"
    assert slack_tools.extract_due_date(item, SCHEMA) == "2026-10-12"


def test_attributive_member_selection_resolves_only_after_member_identity(slack):
    slack.users.append({"id": "UOWNER", "name": "OwnerX"})
    owned = [slack.add(f"Owned work {index}", assignee="UOWNER") for index in range(3)]
    response = ask("Assign the second OwnerX task to Morgan", thread="ATTRIBUTIVE_MEMBER")
    assert "Action item updated" in response
    assert slack_tools.extract_assignee_ids(owned[1], SCHEMA) == ["UM"]
    assert slack_tools.extract_assignee_ids(owned[0], SCHEMA) == ["UOWNER"]


def test_unresolved_tentative_member_falls_back_to_exact_task_title(slack):
    titled = slack.add("first review")
    response = ask("Delete first review task", thread="TITLE_NOT_MEMBER")
    assert "deletion verified" in response
    assert titled not in slack.items


def test_member_title_ambiguity_fails_closed(slack):
    slack.users.append({"id": "UREVIEW", "name": "review"})
    slack.add("first review")
    slack.add("Review owned work", assignee="UREVIEW")
    response = ask("Delete first review task", thread="MEMBER_TITLE_AMBIGUITY")
    assert "could mean an exact task title" in response
    assert len(slack.items) == 2
    assert not slack.writes


def test_counted_possessive_selection_filters_then_selects(slack):
    slack.users.append({"id": "UOWNER", "name": "OwnerX"})
    owned = [slack.add(f"Owned {index}", assignee="UOWNER") for index in range(4)]
    response = ask("Move OwnerX's last two pending tasks to me", thread="POSSESSIVE_COUNT")
    assert "Action items updated" in response
    assert [slack_tools.extract_assignee_ids(item, SCHEMA) for item in owned] == [
        ["UOWNER"], ["UOWNER"], ["UA"], ["UA"]]


def test_central_slack_formatter_normalizes_bold_markdown(monkeypatch):
    sent = {}
    client = SimpleNamespace(chat_postMessage=lambda **kwargs: sent.update(kwargs) or {"ok": True})
    monkeypatch.setattr(main, "app", SimpleNamespace(client=client))
    main.post("C", r"**First task** and \*\*Second task\*\*")
    assert sent["text"] == "*First task* and *Second task*"
    assert "**" not in sent["text"]


@pytest.mark.parametrize("phrase", [
    "delete first docs task and web task from @Praveen",
    "delete the first docs task and web task assigned to Praveen",
    "remove Praveen's first docs and web tasks",
    "delete the first matching docs task and web task from Praveen",
])
def test_grouped_task_constraints_preserve_filters_and_per_group_selection(phrase):
    parsed = commands.validate_command(intent_parser.parse_intent(phrase))
    assert parsed["intent"] == "delete"
    assert [group["query"] for group in parsed["target_groups"]] == ["docs", "web"]
    assert all(group["reference"]["positions"] == (1,) for group in parsed["target_groups"])
    assert parsed["assignees"] in (["Praveen"], ["@Praveen"])


def test_grouped_selection_resolves_first_exact_id_from_each_filtered_group(slack, monkeypatch):
    docs = [slack.add("docs alpha", assignee="UP"), slack.add("docs beta", assignee="UP")]
    web = [slack.add("web alpha", assignee="UP"), slack.add("web beta", assignee="UP")]
    other = slack.add("docs other owner", assignee="UM")
    captured = []
    original = mutations.execute_collection

    def record(item_ids, intent, changes, ctx, schema):
        captured.extend(item_ids)
        return original(item_ids, intent, changes, ctx, schema)

    monkeypatch.setattr(mutations, "execute_collection", record)
    response = ask("delete first docs task and web task from @Praveen", thread="GROUPED_DELETE")
    assert "Action items deleted" in response
    assert captured == [docs[0]["id"], web[0]["id"]]
    assert slack.items == [docs[1], web[1], other]


def test_grouped_constraints_support_independent_second_and_last_selection(slack):
    docs = [slack.add(f"docs {index}", assignee="UP") for index in range(3)]
    web = [slack.add(f"web {index}", assignee="UP") for index in range(3)]
    response = ask(
        "delete second docs task and last web task assigned to Praveen",
        thread="GROUPED_POSITIONS")
    assert "Action items deleted" in response
    remaining_ids = {item["id"] for item in slack.items}
    assert docs[1]["id"] not in remaining_ids
    assert web[-1]["id"] not in remaining_ids


@pytest.mark.parametrize("selector", ["both", "all"])
def test_grouped_collection_selectors_apply_within_each_filtered_group(slack, selector):
    docs = [slack.add(f"docs {index}", assignee="UP") for index in range(2)]
    web = [slack.add(f"web {index}", assignee="UP") for index in range(2)]
    unrelated = slack.add("docs other owner", assignee="UM")
    response = ask(
        f"delete {selector} docs tasks and {selector} web tasks from Praveen",
        thread=f"GROUPED_{selector.upper()}")
    assert "Action items deleted" in response
    removed = {item["id"] for item in docs + web}
    assert removed.isdisjoint({item["id"] for item in slack.items})
    assert slack.items == [unrelated]


def test_grouped_assignment_updates_existing_exact_ids_and_preserves_other_fields(slack, monkeypatch):
    docs = slack.add("docs alpha", assignee="UP", priority="P1", due="2026-10-01")
    web = slack.add("web alpha", assignee="UP", priority="P3", due="2026-10-02")
    original_ids = [docs["id"], web["id"]]
    captured = []
    original = mutations.execute_collection

    def record(item_ids, intent, changes, ctx, schema):
        captured.extend(item_ids)
        return original(item_ids, intent, changes, ctx, schema)

    monkeypatch.setattr(mutations, "execute_collection", record)
    response = ask(
        "assign first docs task and web task from Praveen to Morgan",
        thread="GROUPED_ASSIGN")
    assert "Action items updated" in response
    assert captured == original_ids
    assert [item["id"] for item in slack.items] == original_ids
    assert [slack_tools.extract_assignee_ids(item, SCHEMA) for item in slack.items] == [["UM"], ["UM"]]
    assert [slack_tools.extract_priority(item, SCHEMA) for item in slack.items] == ["P1", "P3"]
    assert [slack_tools.extract_due_date(item, SCHEMA) for item in slack.items] == ["2026-10-01", "2026-10-02"]


def test_grouped_constraints_without_selection_fail_on_duplicate_matches(slack):
    for name in ("docs alpha", "docs beta", "web alpha", "web beta"):
        slack.add(name, assignee="UP")
    response = ask("delete docs task and web task from Praveen", thread="GROUPED_AMBIGUOUS")
    assert "matches 2 items" in response
    assert not slack.writes


def test_grouped_delete_still_enforces_member_rbac(slack):
    slack.add("docs work", assignee="UM")
    slack.add("web work", assignee="UM")
    response = ask("delete first docs task and web task from Morgan", user="UM", thread="GROUPED_RBAC")
    assert "Permission denied" in response
    assert len(slack.items) == 2
    assert not slack.writes


def test_grouped_target_operation_can_coexist_with_independent_operation():
    parsed = commands.validate_command(intent_parser.parse_intent(
        "delete first docs task and web task from Praveen then reopen Release review"))
    assert parsed["intent"] == "compound"
    assert len(parsed["operations"]) == 2
    assert len(parsed["operations"][0]["target_groups"]) == 2
    assert parsed["operations"][1]["intent"] == "reopen"


@pytest.mark.parametrize("phrase", [
    "give Release notes to Aastha",
    "make Aastha responsible for Release notes",
    "Aastha should handle Release notes",
])
def test_assignment_wordings_share_structure(phrase):
    parsed = commands.validate_command(intent_parser.parse_intent(phrase))
    assert parsed["intent"] == "update"
    assert parsed["task_name"] == "Release notes"
    assert parsed["changes"] == [{"field": "assignee", "value": "Aastha"}]


@pytest.mark.parametrize("phrase", [
    "move Release notes from Aastha to Praveen",
    "reassign Release notes to Praveen",
    "transfer Release notes to Praveen",
])
def test_reassignment_wordings_share_structure(phrase):
    parsed = commands.validate_command(intent_parser.parse_intent(phrase))
    assert parsed["intent"] == "update"
    assert parsed["task_name"] == "Release notes"
    assert parsed["changes"] == [{"field": "assignee", "value": "Praveen"}]


@pytest.mark.parametrize("phrase", [
    "show Aastha's tasks",
    "show tasks belonging to Aastha",
    "find work assigned to Aastha",
])
def test_single_assignee_filter_wordings_share_structure(phrase):
    parsed = commands.validate_command(intent_parser.parse_intent(phrase))
    assert parsed["intent"] == "list"
    assert parsed["assignees"] == ["Aastha"]
    assert parsed["completed"] is False


@pytest.mark.parametrize("phrase", [
    "show tasks assigned to Aastha and Praveen",
    "list work belonging to Aastha & Praveen",
    "find tasks of Aastha, Praveen",
])
def test_multiple_assignee_filters_share_or_semantics(phrase):
    parsed = commands.validate_command(intent_parser.parse_intent(phrase))
    assert parsed["intent"] == "list"
    assert parsed["assignees"] == ["Aastha", "Praveen"]


@pytest.mark.parametrize("phrase", [
    "delete all the tasks of Aastha and Praveen",
    "remove everything assigned to Aastha and Praveen",
    "clear every task belonging to Aastha & Praveen",
])
def test_bulk_delete_wordings_share_structure(phrase):
    parsed = commands.validate_command(intent_parser.parse_intent(phrase))
    assert parsed["intent"] == "delete"
    assert parsed["reference"]["kind"] == "all"
    assert parsed["reference_scope"] == "filtered"
    assert parsed["assignees"] == ["Aastha", "Praveen"]


def test_bulk_multiple_assignee_execution_uses_or_filter(slack):
    aastha = slack.add("Aastha work", assignee="UAA")
    praveen = slack.add("Praveen work", assignee="UP")
    other = slack.add("Other work", assignee="UM")
    response = ask("remove everything assigned to Aastha and Praveen")
    assert "deleted" in response
    assert slack.items == [other]
    deleted = [payload["id"] for method, payload in slack.writes if method.endswith("delete")]
    assert deleted == [aastha["id"], praveen["id"]]


@pytest.mark.parametrize("phrase", [
    "delete all the items",
    "delete all action items",
    "delete all tasks",
    "remove every current action item",
])
def test_all_applicable_collection_resolves_live_exact_ids_without_context(slack, monkeypatch, phrase):
    items = [slack.add("Alpha"), slack.add("Beta"), slack.add("Gamma")]
    captured = []
    original = mutations.execute_collection
    def recording_execute(item_ids, intent, changes, ctx, schema):
        captured.extend(item_ids)
        return original(item_ids, intent, changes, ctx, schema)
    monkeypatch.setattr(mutations, "execute_collection", recording_execute)
    response = ask(phrase, thread="NEW_WITH_NO_DISPLAY")
    assert "Action items deleted" in response
    assert captured == [item["id"] for item in items]
    assert slack.items == []


def test_target_resolver_distinguishes_live_and_contextual_collections(slack):
    items = [slack.add("Alpha"), slack.add("Beta")]
    ctx = main.context("UA", "C", "T", None, "W")
    live_command = commands.validate_command(intent_parser.parse_intent("delete all tasks"))
    live = main.resolve_target_set(live_command, deepcopy(slack.items), SCHEMA, "unused", ctx, "delete")
    assert live.target_type is TargetType.ALL_APPLICABLE_ITEMS
    assert list(live.item_ids) == [item["id"] for item in items]
    main.store_view(main.context_keys(ctx)[0], [items[1]], SCHEMA, ctx)
    contextual = main.resolve_target_set(
        commands.validate_command(intent_parser.parse_intent("delete all")),
        deepcopy(slack.items), SCHEMA, "unused", ctx, "delete")
    assert contextual.target_type is TargetType.CONTEXTUAL_ITEMS
    assert list(contextual.item_ids) == [items[1]["id"]]


def test_slack_workspace_admin_metadata_does_not_bypass_configured_role(slack, monkeypatch):
    slack.users.append({"id": "UADMIN", "name": "Workspace Admin", "is_admin": True})
    monkeypatch.setattr(config, "USER_ROLES", {})
    monkeypatch.setattr(config, "DEFAULT_ROLE", "viewer")
    ctx = main.context("UADMIN", "C", "T", None, "W")
    assert ctx.role == "viewer"
    assert not config.has_permission(ctx, "delete")


def test_admin_complete_all_applicable_items_is_verified(slack):
    items = [slack.add("Alpha"), slack.add("Beta", assignee="UM")]
    response = ask("complete all action items")
    assert "Action items completed" in response
    assert all(slack_tools.extract_completed(item, SCHEMA) for item in items)


def test_admin_bulk_update_and_reassignment_use_exact_collections(slack):
    items = [slack.add("Alpha", assignee="UM"), slack.add("Beta", assignee="UAA")]
    assert "Action items updated" in ask("change priority of all action items to P1")
    assert all(slack_tools.extract_priority(item, SCHEMA) == "P1" for item in items)
    assert "Action items updated" in ask("assign all pending tasks to Morgan")
    assert all(slack_tools.extract_assignee_ids(item, SCHEMA) == ["UM"] for item in items)


def test_member_cannot_delete_collection_owned_by_other_members(slack):
    slack.add("Admin work", assignee="UA")
    slack.add("Other work", assignee="UAA")
    response = ask("delete all action items", user="UM")
    assert "Permission denied" in response
    assert not slack.writes


def test_bulk_resolution_combines_member_status_priority_and_due_filters(slack):
    in_week = (main.current_date() + timedelta(days=2)).isoformat()
    later = (main.current_date() + timedelta(days=20)).isoformat()
    wanted_a = slack.add("Aastha urgent", assignee="UAA", priority="P1", due=in_week)
    wanted_p = slack.add("Praveen urgent", assignee="UP", priority="P1", due=in_week)
    slack.add("Wrong member", assignee="UM", priority="P1", due=in_week)
    slack.add("Wrong priority", assignee="UAA", priority="P2", due=in_week)
    slack.add("Wrong date", assignee="UP", priority="P1", due=later)
    slack.add("Already done", completed=True, assignee="UAA", priority="P1", due=in_week)
    command = commands.validate_command({
        "intent": "complete", "assignees": ["Aastha", "Praveen"],
        "priority": "P1", "completed": False, "due_this_week": True,
        "reference": {"kind": "all", "positions": [], "count": 0},
        "reference_scope": "filtered",
    })
    ctx = main.context("UA", "C", "T", None, "W")
    command = main._resolve_command_members(command, ctx)
    response = main._dispatch(command, ctx, "unused")
    assert "Action items completed" in response
    changed = [payload["cells"][0]["row_id"] for method, payload in slack.writes if method.endswith("update")]
    assert changed == [wanted_a["id"], wanted_p["id"]]


def test_read_and_bulk_resolution_share_query_filter_semantics(slack):
    matching = slack.add("Release security review", assignee="UAA")
    slack.add("Release documentation", assignee="UAA")
    slack.add("Security review", assignee="UM")
    parsed = {"intent": "delete", "query": "security", "assignees": ["Aastha"],
              "reference": {"kind": "all", "positions": [], "count": 0},
              "reference_scope": "filtered"}
    response = main._dispatch(main._resolve_command_members(commands.validate_command(parsed),
                                                             main.context("UA", "C", "T", None, "W")),
                              main.context("UA", "C", "T", None, "W"), "unused")
    assert "deletion verified" in response
    assert matching not in slack.items and len(slack.items) == 2


def test_multi_user_list_filter_returns_union(slack):
    slack.add("Aastha work", assignee="UAA")
    slack.add("Praveen work", assignee="UP")
    slack.add("Other work", assignee="UM")
    response = ask("show tasks assigned to Aastha and Praveen")
    assert "Aastha work" in response and "Praveen work" in response
    assert "Other work" not in response


def test_contextual_user_reference_reuses_previous_filter(slack):
    slack.add("Aastha work", assignee="UAA")
    slack.add("Praveen work", assignee="UP")
    slack.add("Other work", assignee="UM")
    ask("show tasks assigned to Aastha and Praveen")
    response = ask("delete all of their tasks")
    assert "deleted" in response
    assert [slack_tools.extract_item_name(item, SCHEMA) for item in slack.items] == ["Other work"]


def test_assign_multiple_users_preserves_all_ids(slack):
    item = slack.add("Release notes")
    response = ask("give Release notes to Aastha and Praveen")
    assert "updated" in response
    assert slack_tools.extract_assignee_ids(item, SCHEMA) == ["UAA", "UP"]


def test_assign_multiple_tasks_to_multiple_users(slack):
    items = [slack.add("Alpha"), slack.add("Beta"), slack.add("Gamma")]
    ask("show tasks")
    response = ask("assign the first two to Aastha and Praveen")
    assert "updated" in response
    assert slack_tools.extract_assignee_ids(items[0], SCHEMA) == ["UAA", "UP"]
    assert slack_tools.extract_assignee_ids(items[1], SCHEMA) == ["UAA", "UP"]
    assert slack_tools.extract_assignee_ids(items[2], SCHEMA) == []


def test_untrusted_model_cannot_supply_resolved_member_ids():
    parsed = commands.validate_command({"intent": "list", "assignees": ["Aastha"],
                                        "resolved_assignee_ids": ["ATTACKER"]})
    assert "resolved_assignee_ids" not in parsed


def test_duplicate_member_clarification_preserves_selected_id(slack):
    slack.users.extend([{"id": "US1", "name": "Sam"}, {"id": "US2", "name": "Sam"}])
    first = slack.add("First Sam", assignee="US1")
    second = slack.add("Second Sam", assignee="US2")
    response = ask("show Sam's tasks")
    assert "Which Slack member" in response
    response = ask("the second one")
    assert "Second Sam" in response and "First Sam" not in response
    state = main._state(main.context("UA", "C", "T", None, "W"))
    assert state["query_filter"]["resolved_assignee_ids"] == ["US2"]
    assert first in slack.items and second in slack.items


def test_workspace_member_query_uses_slack_identity_and_dynamic_role(slack):
    parsed = commands.validate_command({"intent": "members", "members": ["Aastha"]})
    ctx = main.context("UA", "C", "T", None, "W")
    response = main._dispatch(main._resolve_command_members(parsed, ctx), ctx, "unused")
    assert "Aastha" in response and "<@UAA>" in response
    assert "viewer" in response


def test_workspace_role_filter_uses_configured_roles(slack, monkeypatch):
    monkeypatch.setattr(config, "USER_ROLES", {"W:UAA": "manager", "W:UP": "member", "UA": "admin"})
    ctx = main.context("UA", "C", "T", None, "W")
    response = main._dispatch(commands.validate_command({"intent": "members", "role": "manager"}), ctx, "unused")
    assert "Aastha" in response
    assert "Praveen" not in response


def test_member_intent_model_output_is_structured_and_validated(monkeypatch):
    import json
    import langchain_ollama
    monkeypatch.setattr(intent_parser, "OLLAMA_API_KEY", "unit-test-placeholder")
    model_result = {"intent": "members", "members": ["Morgan"]}
    monkeypatch.setattr(langchain_ollama, "ChatOllama", lambda **kwargs: SimpleNamespace(
        invoke=lambda messages: SimpleNamespace(content=json.dumps(model_result))))
    parsed = commands.validate_command(intent_parser.parse_intent("Could you identify the workspace member Morgan?"))
    assert parsed["intent"] == "members"
    assert parsed["members"] == ["Morgan"]


def test_compound_operations_execute_in_order_with_exact_targets(slack):
    item = slack.add("Release notes")
    command = commands.validate_command({"intent": "compound", "operations": [
        {"intent": "update", "task_name": "Release notes",
         "changes": [{"field": "priority", "value": "P1"}]},
        {"intent": "complete", "task_name": "Release notes"},
    ]})
    command = main._resolve_command_members(command, main.context("UA", "C", "T", None, "W"))
    response = main._dispatch(command, main.context("UA", "C", "T", None, "W"), "unused")
    assert "updated" in response and "completed" in response
    assert slack_tools.extract_priority(item, SCHEMA) == "P1"
    assert slack_tools.extract_completed(item, SCHEMA)


def test_compound_command_validation_rejects_nesting():
    with pytest.raises(ValueError):
        commands.validate_command({"intent": "compound", "operations": [
            {"intent": "compound", "operations": []}, {"intent": "list"}]})


def test_batch_metadata_overrides_and_past_date(slack):
    tomorrow = (main.current_date() + timedelta(days=1)).isoformat()
    past = (main.current_date() - timedelta(days=1)).isoformat()
    result = ask(f"Create P2 tasks due {tomorrow}:\n- Alpha priority P1 for Morgan\n- Beta due {past}\n- Gamma")
    assert "past" in result
    assert len(slack.items) == 2
    alpha, gamma = slack.items
    assert slack_tools.extract_item_name(alpha, SCHEMA) == "Alpha"
    assert slack_tools.extract_priority(alpha, SCHEMA) == "P1"
    assert slack_tools.extract_assignee_id(alpha, SCHEMA) == "UM"
    assert slack_tools.extract_priority(gamma, SCHEMA) == "P2"
    assert slack_tools.extract_assignee_id(gamma, SCHEMA) is None
    assert slack_tools.extract_due_date(gamma, SCHEMA) == tomorrow


def test_invalid_explicit_date_is_not_silently_dropped(slack):
    response = ask("Create Alpha due February 30, 2027")
    assert "created successfully" not in response
    assert not slack.writes


def test_three_tasks_in_one_sentence(slack):
    result = ask("Create Alpha and Beta and Gamma")
    assert len(slack.items) == 3
    assert "could not" not in result


def test_quoted_title_with_conjunction_and_metadata_words(slack):
    ask('Create a task called "Research and Development P1"')
    assert len(slack.items) == 1
    assert slack_tools.extract_item_name(slack.items[0], SCHEMA) == "Research and Development P1"
    assert not slack_tools.extract_priority(slack.items[0], SCHEMA)


def test_unseen_syntactic_variation(slack):
    items = [slack.add("Alpha"), slack.add("Beta")]
    ask("show tasks")
    response = ask("I'd like you to mark the final entry on that displayed list finished.")
    assert "completed" in response
    assert slack.writes[-1][1]["cells"][0]["row_id"] == items[1]["id"]


def test_ambiguous_clarification_survives_restart(slack):
    slack.add("Report alpha")
    target = slack.add("Report beta")
    ask("delete Report")
    main._pending.clear()
    assert "deleted" in ask("second")
    assert target not in slack.items


def test_clarification_missing_candidate_does_not_pick_another(slack):
    first = slack.add("Report")
    second = slack.add("Report")
    ask("delete Report")
    slack.items.remove(second)
    assert "no longer exists" in ask("second")
    assert slack.items == [first]
    assert not slack.writes


def test_interrupted_mutation_resumes_same_id_without_duplicate_write(slack, monkeypatch):
    first, last = slack.add("Alpha"), slack.add("Beta")
    ask("show tasks")
    original_save = main._save_state
    def failed_save(*args):
        raise RuntimeError("simulated process interruption after write")
    monkeypatch.setattr(main, "_save_state", failed_save)
    send = Mock(return_value={"ts": "B"})
    with pytest.raises(RuntimeError):
        delivery.execute_event(main._db, "resume-mutation", lambda: ask("complete last"), send)
    monkeypatch.setattr(main, "_save_state", original_save)
    # A later list and reordered live data must not change the saved plan.
    slack.items.reverse()
    ask("show all")
    delivery.execute_event(main._db, "resume-mutation", lambda: ask("complete last"), send)
    assert len(slack.writes) == 1
    assert slack_tools.extract_completed(last, SCHEMA)
    assert not slack_tools.extract_completed(first, SCHEMA)


def test_interrupted_create_resumes_exact_created_id(slack, monkeypatch):
    save = main._save_state
    monkeypatch.setattr(main, "_save_state", Mock(side_effect=RuntimeError("interruption")))
    send = Mock(return_value={"ts": "B"})
    with pytest.raises(RuntimeError):
        delivery.execute_event(main._db, "resume-create", lambda: ask("create Alpha"), send)
    monkeypatch.setattr(main, "_save_state", save)
    delivery.execute_event(main._db, "resume-create", lambda: ask("create Alpha"), send)
    assert len(slack.items) == 1
    assert len(slack.writes) == 1


def test_uncertain_create_without_id_is_not_repeated(slack):
    with delivery.event(main._db, "interrupted-before-id"):
        ctx = config.build_context("UA", "C", team_id="W", thread_ts="T")
        expected = [{"field": "name", "value": "Alpha"}, {"field": "completed", "value": False}]
        key = delivery.checkpoint_key("create", {"list": "L", "assignee": None, "fields": expected})
        delivery.checkpoint_write(key, "started", {"before_ids": []})
        assert "outcome unknown" in main.handle_create({"intent": "create", "task_name": "Alpha"}, ctx)
    assert not slack.writes


def test_missing_update_item_is_not_verified(slack):
    result = mutations.verify("MISSING", [{"field": "priority", "value": "P1"}], config.build_context("UA", "C"), SCHEMA)
    assert not result.verified
    assert "not found" in result.problems[0]


def test_noop_bulk_reports_individual_failure(slack):
    slack.add("Alpha")
    slack.add("Beta")
    ask("show tasks")
    slack.noop = True
    response = ask("complete both")
    assert "not all changes verified" in response
    assert response.count("still shows pending") == 2


@pytest.mark.parametrize("malformed", [
    {"intent": "update", "changes": "priority P1"},
    {"intent": "complete", "reference": {"kind": "positions", "positions": [True]}},
    {"intent": "delete", "selection_numbers": ["all"]},
    {"intent": "create", "task_name": ["Alpha"]},
])
def test_model_data_validation(malformed):
    with pytest.raises(ValueError):
        commands.validate_command(malformed)


@pytest.mark.parametrize("model_result,expected", [
    ({"intent": "out_of_scope"}, "out_of_scope"),
    ({"intent": "update", "task_name": "that task", "changes": [{"field": "priority", "value": "P1"}]}, "update"),
    ({"intent": "complete", "task_name": "Last Client Report", "selection": "last"}, "complete"),
])
def test_model_fallback_validated_and_configured(monkeypatch, model_result, expected):
    import json
    import langchain_ollama
    monkeypatch.setattr(intent_parser, "OLLAMA_API_KEY", "unit-test-placeholder")
    model = Mock(return_value=SimpleNamespace(content=json.dumps(model_result)))
    factory = Mock(return_value=SimpleNamespace(invoke=model))
    monkeypatch.setattr(langchain_ollama, "ChatOllama", factory)
    result = commands.validate_command(intent_parser.parse_intent("Please make the appropriate adjustment to my action item."))
    assert result["intent"] == expected
    assert "Authorization" in factory.call_args.kwargs["client_kwargs"]["headers"]
    if model_result.get("task_name") == "Last Client Report":
        assert result.get("selection") is None


def test_distinct_natural_language_actions_use_compound_structure(monkeypatch):
    import json
    import langchain_ollama
    model_result = {"intent": "compound", "operations": [
        {"intent": "update", "task_name": "Release notes",
         "changes": [{"field": "priority", "value": "P1"}]},
        {"intent": "complete", "task_name": "Release notes"},
    ]}
    monkeypatch.setattr(intent_parser, "OLLAMA_API_KEY", "unit-test-placeholder")
    model = Mock(return_value=SimpleNamespace(content=json.dumps(model_result)))
    monkeypatch.setattr(langchain_ollama, "ChatOllama", lambda **kwargs: SimpleNamespace(invoke=model))
    parsed = commands.validate_command(intent_parser.parse_intent(
        "Revise the priority of Release notes to P1, and then finish Release notes."))
    assert parsed["intent"] == "compound"
    assert [operation["intent"] for operation in parsed["operations"]] == ["update", "complete"]


def test_model_cannot_inject_item_ids_or_bypass_role(slack, monkeypatch):
    import json
    import langchain_ollama
    target = slack.add("Alpha")
    wrong = slack.add("Beta")
    monkeypatch.setattr(intent_parser, "OLLAMA_API_KEY", "unit-test-placeholder")
    response = {"intent": "delete", "task_name": "Alpha", "target_ids": [wrong["id"]]}
    monkeypatch.setattr(langchain_ollama, "ChatOllama", lambda **kwargs: SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=json.dumps(response))))
    assert "Permission denied" in ask("Please discard the requested action item.", user="UM")
    assert not slack.writes
    assert "deleted" in ask("Please discard the requested action item.")
    assert slack.items == [wrong]


def test_model_must_not_turn_question_into_mutation(slack, monkeypatch):
    import langchain_ollama
    monkeypatch.setattr(intent_parser, "OLLAMA_API_KEY", "unit-test-placeholder")
    monkeypatch.setattr(langchain_ollama, "ChatOllama", lambda **kwargs: SimpleNamespace(invoke=lambda messages: SimpleNamespace(content='{"intent":"create","task_name":"Invented"}')))
    assert "asking about a task" in ask("Why is the sky blue?")
    assert not slack.writes


@pytest.mark.parametrize("wording", [
    "how many overdue P1 tasks does Morgan have?",
    "count overdue P1 tasks assigned to Morgan",
])
def test_analytical_wording_produces_same_composable_plan(wording):
    parsed = commands.validate_command(intent_parser.parse_intent(wording))
    assert parsed["intent"] == "list"
    assert parsed["aggregate"] == "count"
    assert parsed["overdue"] is True
    assert parsed["priority"] == "P1"
    assert parsed["assignees"] == ["Morgan"]


def test_date_range_is_normalized_and_applied_without_mutation(slack):
    today = main.current_date()
    inside = slack.add("Inside", completed=True, due=(today + timedelta(days=2)).isoformat())
    slack.add("Too late", completed=True, due=(today + timedelta(days=20)).isoformat())
    parsed = commands.validate_command({
        "intent": "list", "completed": True,
        "date_from": (today + timedelta(days=1)).isoformat(),
        "date_to": (today + timedelta(days=7)).isoformat(),
    })
    response = main._dispatch(parsed, main.context("UA", "C", "T", None, "W"), "unused")
    assert slack_tools.extract_item_name(inside, SCHEMA) in response
    assert "Too late" not in response
    assert not slack.writes


def test_sort_limit_and_display_context_keep_exact_ids(slack):
    low = slack.add("Low", priority="P4")
    high = slack.add("High", priority="P1")
    medium = slack.add("Medium", priority="P2")
    response = ask("show the two highest priority pending tasks")
    assert response.index("High") < response.index("Medium")
    assert "Low" not in response
    state = main._state(main.context("UA", "C", "T", None, "W"))
    assert [item["id"] for item in state["items"]] == [high["id"], medium["id"]]
    assert low["id"] not in state["focus_ids"]


def test_grouped_count_executes_after_member_and_status_filters(slack):
    slack.add("Morgan open", assignee="UM")
    slack.add("Morgan done", completed=True, assignee="UM")
    slack.add("Alex open", assignee="UA")
    response = ask("compare the number of open tasks assigned to Morgan and Alex")
    assert "Count by Assignee" in response
    assert "*UM*: 1" in response
    assert "*UA*: 1" in response
    assert not slack.writes


def test_explicit_live_collection_query_is_not_contextual_inspection():
    parsed = commands.validate_command(intent_parser.parse_intent("group all open tasks by assignee"))
    assert parsed["intent"] == "list"
    assert parsed["target_scope"] == "all_applicable"
    assert parsed["group_by"] == "assignee"
    assert parsed.get("reference") is None


@pytest.mark.parametrize("field,value", [
    ("sort_by", "made_up"), ("group_by", "made_up"),
    ("aggregate", "sum"), ("sort_order", "sideways"),
])
def test_untrusted_query_operations_are_validated(field, value):
    with pytest.raises(ValueError):
        commands.validate_command({"intent": "list", field: value})


@pytest.mark.parametrize("wording,expected_index", [
    ("delete the first task assigned to Morgan", 0),
    ("delete the earliest task assigned to Morgan", 0),
    ("delete the latest task assigned to Morgan", 1),
    ("delete the most recent task assigned to Morgan", 1),
])
def test_singular_filtered_selection_mutates_exactly_one_id(slack, wording, expected_index):
    targets = [slack.add("Morgan A", assignee="UM"), slack.add("Morgan B", assignee="UM")]
    slack.add("Other", assignee="UA")
    response = ask(wording)
    assert "Action item deleted" in response
    deleted = [payload["id"] for method, payload in slack.writes if method.endswith("delete")]
    assert deleted == [targets[expected_index]["id"]]


def test_singular_contextual_selection_uses_displayed_ids_not_live_collection(slack):
    first = slack.add("Morgan A", assignee="UM")
    last = slack.add("Morgan B", assignee="UM")
    slack.add("Other", assignee="UA")
    ask("show tasks assigned to Morgan")
    response = ask("delete the most recent task")
    assert "Action item deleted" in response
    assert [payload["id"] for method, payload in slack.writes if method.endswith("delete")] == [last["id"]]
    assert first in slack.items


def test_filtered_collection_still_mutates_every_matching_exact_id(slack):
    targets = [slack.add("Morgan A", assignee="UM"), slack.add("Morgan B", assignee="UM")]
    other = slack.add("Other", assignee="UA")
    response = ask("delete every task assigned to Morgan")
    assert "Action items deleted" in response
    assert [payload["id"] for method, payload in slack.writes if method.endswith("delete")] == [
        item["id"] for item in targets]
    assert slack.items == [other]


def test_normalized_limit_selects_from_filtered_candidates_before_mutation(slack):
    beta = slack.add("Beta", assignee="UM")
    alpha = slack.add("Alpha", assignee="UM")
    slack.add("Aardvark", assignee="UA")
    ctx = main.context("UA", "C", "T", None, "W")
    parsed = commands.validate_command({
        "intent": "delete", "target_scope": "filtered", "assignees": ["Morgan"],
        "sort_by": "name", "sort_order": "asc", "limit": 1,
    })
    parsed = main._resolve_command_members(parsed, ctx)
    response = main._dispatch(parsed, ctx, "unused")
    assert "Action item deleted" in response
    assert [payload["id"] for method, payload in slack.writes if method.endswith("delete")] == [alpha["id"]]
    assert beta in slack.items


def test_mutation_boundary_rejects_singular_expansion(slack):
    first = slack.add("One")
    second = slack.add("Two")
    ctx = main.context("UA", "C", "T", None, "W")
    with pytest.raises(ValueError, match="singular request"):
        main.handle_mutation({
            "intent": "delete", "target_scope": "single",
            "target_ids": [first["id"], second["id"]], "changes": [], "tasks": [],
        }, ctx, "unused")
    assert not slack.writes


def test_persisted_failure_shape_normalizes_filter_and_selection_separately():
    parsed = commands.validate_command(intent_parser.parse_intent(
        "delete recent assigne task of <@UM>"))
    assert parsed["intent"] == "delete"
    assert parsed["assignees"] == ["<@UM>"]
    assert parsed["target_scope"] == "filtered"
    assert parsed["target_selection"] == {
        "mode": "one", "order_by": "created_at", "direction": "desc", "count": 1}
    assert not parsed.get("task_name")


def test_read_filter_sort_select_one_returns_only_exact_record(slack):
    today = main.current_date()
    nearest = slack.add("Nearest", assignee="UM", due=(today + timedelta(days=1)).isoformat())
    slack.add("Later", assignee="UM", due=(today + timedelta(days=5)).isoformat())
    slack.add("Other member", assignee="UA", due=today.isoformat())
    response = ask("show the nearest due task assigned to Morgan")
    assert "Nearest" in response
    assert "Later" not in response and "Other member" not in response
    state = main._state(main.context("UA", "C", "T", None, "W"))
    assert [item["id"] for item in state["items"]] == [nearest["id"]]
    assert not slack.writes


def test_mutation_filter_sort_select_one_excludes_wrong_status(slack):
    completed_high = slack.add("Completed P1", completed=True, assignee="UM", priority="P1")
    selected = slack.add("Open P2", assignee="UM", priority="P2")
    slack.add("Open P3", assignee="UM", priority="P3")
    response = ask("complete the highest priority open task assigned to Morgan")
    assert "Action item completed" in response
    changed = [payload["cells"][0]["row_id"] for method, payload in slack.writes if method.endswith("update")]
    assert changed == [selected["id"]]
    assert completed_high in slack.items


def test_collection_mode_is_not_narrowed_by_candidate_filters(slack):
    targets = [slack.add("One", assignee="UM"), slack.add("Two", assignee="UM")]
    parsed = commands.validate_command(intent_parser.parse_intent(
        "show every pending task assigned to Morgan"))
    assert parsed["target_selection"] == {"mode": "collection"}
    response = ask("show every pending task assigned to Morgan")
    assert all(slack_tools.extract_item_name(item, SCHEMA) in response for item in targets)


@pytest.mark.parametrize("selection", [
    {"mode": "one"},
    {"mode": "one", "order_by": "unknown", "count": 1},
    {"mode": "many", "order_by": "position", "count": 0},
    {"mode": "everything"},
])
def test_untrusted_target_selection_is_validated(selection):
    with pytest.raises(ValueError):
        commands.validate_command({"intent": "list", "target_selection": selection})


def test_qualitative_urgency_is_selection_not_collection(slack):
    selected = slack.add("Urgent", assignee="UM", priority="P1")
    slack.add("Routine", assignee="UM", priority="P3")
    slack.add("Other member", assignee="UA", priority="P1")
    parsed = commands.validate_command(intent_parser.parse_intent(
        "Which of <@UM> tasks is most urgent?"))
    assert parsed["result_operation"] == "select_one"
    assert parsed["target_selection"] == {
        "mode": "one", "order_by": "urgency", "direction": "asc", "count": 1}
    response = ask("Which of <@UM> tasks is most urgent?")
    assert "Urgent" in response
    assert "Routine" not in response and "Other member" not in response
    state = main._state(main.context("UA", "C", "T", None, "W"))
    assert [item["id"] for item in state["items"]] == [selected["id"]]


def test_qualitative_selection_mutation_changes_one_exact_id(slack):
    selected = slack.add("Urgent", assignee="UM", priority="P1")
    slack.add("Routine", assignee="UM", priority="P3")
    response = ask("complete the most urgent open task assigned to Morgan")
    assert "Action item completed" in response
    changed = [payload["cells"][0]["row_id"] for method, payload in slack.writes if method.endswith("update")]
    assert changed == [selected["id"]]


@pytest.mark.parametrize("wording", [
    "show my tasks", "show tasks assigned to me", "show tasks belonging to myself",
])
def test_self_references_resolve_to_requesting_slack_id(slack, wording):
    slack.add("Mine", assignee="UM")
    slack.add("Not mine", assignee="UA")
    response = ask(wording, user="UM")
    assert "Mine" in response and "Not mine" not in response


def test_assignment_change_self_reference_uses_requester_id(slack):
    item = slack.add("Ownership", assignee="UM")
    ask("show tasks")
    response = ask("reassign first to me")
    assert "updated" in response
    assert slack_tools.extract_assignee_ids(item, SCHEMA) == ["UA"]


def test_completed_today_is_not_due_today_and_reports_metadata_limit(slack):
    slack.add("Done but due later", completed=True,
              assignee="UM", due=(main.current_date() + timedelta(days=3)).isoformat())
    parsed = commands.validate_command(intent_parser.parse_intent("What did I finish today?"))
    assert parsed["completed"] is True
    assert parsed["assignee_self"] is True
    assert parsed.get("due_today") is False
    assert parsed["temporal_filter"] == {
        "field": "completed_at", "relation": "on", "date": main.current_date().isoformat()}
    response = ask("What did I finish today?", user="UM")
    assert "not a reliable completion timestamp" in response
    assert "Done but due later" not in response
    assert not slack.writes


def test_due_today_remains_due_date_dimension(slack):
    due = slack.add("Due today", assignee="UM", due=main.current_date().isoformat())
    slack.add("Due later", assignee="UM", due=(main.current_date() + timedelta(days=2)).isoformat())
    parsed = commands.validate_command(intent_parser.parse_intent("What do I need to work on today?"))
    assert parsed["temporal_filter"]["field"] == "due_date"
    response = ask("What do I need to work on today?", user="UM")
    assert slack_tools.extract_item_name(due, SCHEMA) in response
    assert "Due later" not in response


@pytest.mark.parametrize("wording,field", [
    ("show tasks due today", "due_date"),
    ("show tasks created today", "created_at"),
    ("show tasks updated today", "updated_at"),
    ("what tasks were completed today", "completed_at"),
])
def test_temporal_state_and_date_dimensions_remain_distinct(wording, field):
    parsed = commands.validate_command(intent_parser.parse_intent(wording))
    assert parsed["temporal_filter"]["field"] == field
    assert parsed.get("due_today") is (field == "due_date")


def test_created_today_filters_creation_timestamp_not_due_date(slack):
    now = datetime.now().timestamp()
    old = now - (3 * 24 * 60 * 60)
    current = slack.add("Created today", due=(main.current_date() + timedelta(days=5)).isoformat())
    current["date_created"] = now
    older = slack.add("Created earlier", due=main.current_date().isoformat())
    older["date_created"] = old
    response = ask("show tasks created today")
    assert "Created today" in response
    assert "Created earlier" not in response


def test_workspace_scoped_role_and_permissions_are_dynamic(slack, monkeypatch):
    slack.add("Alpha")
    monkeypatch.setattr(config, "USER_ROLES", {"W:UX": "member"})
    monkeypatch.setitem(config.PERMISSIONS, "W:member", {"view"})
    ctx = main.context("UX", "C", "T", None, "W")
    assert ctx.role == "member"
    assert config.has_permission(ctx, "view")
    assert not config.has_permission(ctx, "complete")
    assert "Permission denied" in ask("complete Alpha", user="UX")
    assert not slack.writes


def test_reopen_and_complete_permissions_are_independent(slack, monkeypatch):
    slack.add("Alpha", completed=True, assignee="UM")
    monkeypatch.setitem(config.PERMISSIONS, "member", {"view", "reopen", "edit_status"})
    assert "reopened" in ask("reopen Alpha", user="UM")
    assert "Permission denied" in ask("complete Alpha", user="UM")


def test_bulk_permission_is_centralized_and_prevents_all_writes(slack, monkeypatch):
    slack.add("Alpha", assignee="UM")
    slack.add("Beta", assignee="UM")
    monkeypatch.setitem(config.PERMISSIONS, "member", {"view", "complete", "edit_status"})
    ask("show tasks", user="UM")
    response = ask("complete all", user="UM")
    assert "bulk" in response.lower()
    assert not slack.writes


def test_assign_and_reassign_permissions_use_current_item_state(slack, monkeypatch):
    item = slack.add("Alpha")
    monkeypatch.setattr(config, "USER_ROLES", {"UA": "admin", "UX": "manager"})
    monkeypatch.setitem(config.PERMISSIONS, "manager", {"view", "update", "update_others", "assign", "edit_assignee"})
    ctx = main.context("UX", "C", "T", None, "W")
    first = {"intent": "update", "task_name": "Alpha", "tasks": [],
             "changes": [{"field": "assignee", "value": "Alex"}]}
    assert "updated" in main.handle_mutation(first, ctx, "unused")
    assert slack_tools.extract_assignee_ids(item, SCHEMA) == ["UA"]
    second = {"intent": "update", "task_name": "Alpha", "tasks": [],
              "changes": [{"field": "assignee", "value": "Praveen"}]}
    with pytest.raises(PermissionError, match="reassign"):
        main.handle_mutation(second, ctx, "unused")
    assert slack_tools.extract_assignee_ids(item, SCHEMA) == ["UA"]


def test_list_specific_field_controls_apply_to_read_and_update(slack, monkeypatch):
    slack.add("Alpha", priority="P1")
    monkeypatch.setitem(config.FIELD_CONTROLS, "L:priority",
                        {"read": "read_restricted_priority", "edit": "edit_restricted_priority"})
    response = ask("show tasks")
    assert "Alpha" in response and "Priority" not in response
    ctx = main.context("UA", "C", "T", None, "W")
    with pytest.raises(PermissionError, match="priority"):
        main.handle_mutation({"intent": "update", "task_name": "Alpha", "tasks": [],
                              "changes": [{"field": "priority", "value": "P2"}]}, ctx, "unused")
    assert not slack.writes


def test_single_field_update_preserves_unrelated_dynamic_cells(slack):
    item = slack.add("Alpha", assignee="UM", priority="P1",
                     due=(main.current_date() + timedelta(days=1)).isoformat())
    before = {field["column_id"]: deepcopy(field) for field in item["fields"]}
    ctx = main.context("UA", "C", "T", None, "W")
    new_due = (main.current_date() + timedelta(days=4)).isoformat()
    result = main.handle_mutation({"intent": "update", "task_name": "Alpha", "tasks": [],
                                   "changes": [{"field": "due_date", "value": new_due}]}, ctx, "unused")
    assert "updated" in result
    after = {field["column_id"]: field for field in item["fields"]}
    for column_id in {"name", "owner", "priority", "done"}:
        assert after[column_id] == before[column_id]
    assert slack_tools.extract_due_date(item, SCHEMA) == new_due


def test_team_and_channel_resolve_their_own_lists(slack, monkeypatch):
    monkeypatch.setattr(config, "CHANNEL_LISTS", {
        "W:C_ONE": "LIST_ONE", "W:C_TWO": "LIST_TWO",
    })
    monkeypatch.setattr(config, "DEFAULT_LIST_ID", "")
    slack.add("Alpha")
    ask("show tasks", channel="C_ONE", team="W")
    ask("show tasks", channel="C_TWO", team="W")
    assert "LIST_ONE" in slack.list_requests
    assert "LIST_TWO" in slack.list_requests


def test_channel_list_context_never_crosses_channels(slack, monkeypatch):
    monkeypatch.setattr(config, "CHANNEL_LISTS", {"C_ONE": "LIST_ONE", "C_TWO": "LIST_TWO"})
    monkeypatch.setattr(config, "DEFAULT_LIST_ID", "")
    slack.add("Alpha")
    ask("show tasks", channel="C_ONE")
    response = ask("complete first", channel="C_TWO")
    assert "recently displayed" in response
    assert not slack.writes


def test_unmapped_channel_fails_closed_without_explicit_default(slack, monkeypatch):
    monkeypatch.setattr(config, "CHANNEL_LISTS", {})
    monkeypatch.setattr(config, "DEFAULT_LIST_ID", "")
    response = ask("show tasks", channel="UNMAPPED")
    assert "not mapped" in response
    assert not slack.list_requests


@pytest.mark.parametrize("default_role,mapping,user_id,expected", [
    ("admin", {"U_MEMBER": "member"}, "U_MEMBER", "member"),
    ("member", {"U_MANAGER": "manager"}, "U_MANAGER", "manager"),
    ("member", {"U_ADMIN": "admin"}, "U_ADMIN", "admin"),
    ("member", {}, "U_UNMAPPED", "member"),
    ("admin", {}, "U_UNMAPPED_WITH_EXPLICIT_ADMIN_DEFAULT", "admin"),
    ("viewer", {"U_ONE": "member", "U_TWO": "manager", "U_THREE": "admin"}, "U_TWO", "manager"),
])
def test_role_resolution_explicit_mapping_precedes_configured_default(
        monkeypatch, default_role, mapping, user_id, expected):
    monkeypatch.setattr(config, "DEFAULT_ROLE", default_role)
    monkeypatch.setattr(config, "USER_ROLES", mapping)
    assert config.get_user_role(user_id, "WORKSPACE") == expected


def test_team_scoped_role_precedes_direct_mapping_and_default(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_ROLE", "viewer")
    monkeypatch.setattr(config, "USER_ROLES", {
        "U_DYNAMIC": "member", "TEAM_A:U_DYNAMIC": "manager", "TEAM_B:U_DYNAMIC": "admin",
    })
    assert config.get_user_role("U_DYNAMIC", "TEAM_A") == "manager"
    assert config.get_user_role("U_DYNAMIC", "TEAM_B") == "admin"
    assert config.get_user_role("U_DYNAMIC", "TEAM_C") == "member"


def test_member_is_limited_to_own_tasks_for_reads_and_mutations(slack):
    own = slack.add("Own work", assignee="UM")
    other = slack.add("Other work", assignee="UA")

    response = ask("show all tasks", user="UM")
    assert "Own work" in response
    assert "Other work" not in response
    assert "Permission denied" in ask("show tasks assigned to Alex", user="UM")
    assert "Permission denied" in ask("complete Other work", user="UM")
    assert "Permission denied" in ask("delete Own work", user="UM")
    assert not slack_tools.extract_completed(own, SCHEMA)
    assert not slack_tools.extract_completed(other, SCHEMA)
    assert not slack.writes


def test_member_creates_for_self_and_updates_own_permitted_fields(slack):
    ctx = main.context("UM", "C", "T", None, "W")
    response = main.handle_create({"intent": "create", "task_name": "Personal work"}, ctx)
    assert "created successfully" in response
    item = slack.items[0]
    assert slack_tools.extract_assignee_ids(item, SCHEMA) == ["UM"]

    response = main.handle_mutation({
        "intent": "update", "task_name": "Personal work", "tasks": [],
        "changes": [{"field": "name", "value": "Renamed personal work"}],
    }, ctx, "unused")
    assert "updated" in response
    assert slack_tools.extract_item_name(item, SCHEMA) == "Renamed personal work"


def test_member_cannot_create_for_another_user_but_manager_can(slack, monkeypatch):
    member_ctx = main.context("UM", "C", "MEMBER_CREATE", None, "W")
    with pytest.raises(PermissionError, match="assigned to you"):
        main.handle_create({
            "intent": "create", "task_name": "Delegated by member",
            "resolved_assignee_ids": ["UA"],
        }, member_ctx)
    assert not slack.writes

    monkeypatch.setattr(config, "USER_ROLES", {"U_MANAGER": "manager"})
    manager_ctx = main.context("U_MANAGER", "C", "MANAGER_CREATE", None, "W")
    response = main.handle_create({
        "intent": "create", "task_name": "Delegated by manager",
        "resolved_assignee_ids": ["UM"],
    }, manager_ctx)
    assert "created successfully" in response
    assert slack_tools.extract_assignee_ids(slack.items[0], SCHEMA) == ["UM"]


def test_member_bulk_permission_cannot_expand_ownership_scope(slack, monkeypatch):
    own = slack.add("Own work", assignee="UM")
    other = slack.add("Other work", assignee="UA")
    monkeypatch.setitem(config.PERMISSIONS, "member",
                        config.PERMISSIONS["member"] | {"bulk"})
    ctx = main.context("UM", "C", "T", None, "W")
    with pytest.raises(PermissionError, match="assigned to you"):
        mutations.authorize_collection(
            [own["id"], other["id"]], deepcopy(slack.items), "complete",
            [{"field": "completed", "value": True}], ctx, SCHEMA)
    assert not slack.writes


def test_manager_can_operate_across_users_but_cannot_delete(slack, monkeypatch):
    item = slack.add("Other user's work", assignee="UM")
    monkeypatch.setattr(config, "USER_ROLES", {"U_MANAGER": "manager", "UM": "member"})
    ctx = main.context("U_MANAGER", "C", "T", None, "W")

    assert "Other user's work" in main.handle_inspect(
        {"intent": "inspect", "task_name": "Other user's work"}, ctx, "unused")
    assert "updated" in main.handle_mutation({
        "intent": "update", "task_name": "Other user's work", "tasks": [],
        "changes": [{"field": "assignee", "value": "Alex"}],
    }, ctx, "unused")
    assert slack_tools.extract_assignee_ids(item, SCHEMA) == ["UA"]
    with pytest.raises(PermissionError, match="delete"):
        main.handle_mutation({"intent": "delete", "task_name": "Other user's work", "tasks": []}, ctx, "unused")
    assert item in slack.items


def test_manager_bulk_updates_across_users_and_admin_can_delete(slack, monkeypatch):
    first = slack.add("First", assignee="UM")
    second = slack.add("Second", assignee="UAA")
    monkeypatch.setattr(config, "USER_ROLES", {"U_MANAGER": "manager", "U_ADMIN": "admin"})

    assert "Action items completed" in ask("complete all action items", user="U_MANAGER")
    assert all(slack_tools.extract_completed(item, SCHEMA) for item in (first, second))
    assert "Action items deleted" in ask("delete all action items", user="U_ADMIN", thread="ADMIN")
    assert slack.items == []


def test_field_permissions_apply_after_ownership_and_preserve_other_cells(slack, monkeypatch):
    item = slack.add("Own work", assignee="UM", priority="P1",
                     due=(main.current_date() + timedelta(days=1)).isoformat())
    before = {field["column_id"]: deepcopy(field) for field in item["fields"]}
    monkeypatch.setitem(config.PERMISSIONS, "member",
                        {"view", "update", "edit_name", "edit_due_date"})
    ctx = main.context("UM", "C", "T", None, "W")

    with pytest.raises(PermissionError, match="priority"):
        main.handle_mutation({
            "intent": "update", "task_name": "Own work", "tasks": [],
            "changes": [{"field": "priority", "value": "P2"}],
        }, ctx, "unused")
    assert "updated" in main.handle_mutation({
        "intent": "update", "task_name": "Own work", "tasks": [],
        "changes": [{"field": "name", "value": "Renamed own work"}],
    }, ctx, "unused")
    after = {field["column_id"]: field for field in item["fields"]}
    assert after["owner"] == before["owner"]
    assert after["priority"] == before["priority"]
    assert after["due"] == before["due"]
    assert after["done"] == before["done"]


def test_authorization_logic_contains_no_user_specific_identity_rules():
    sources = "\n".join(path.read_text() for path in (
        __import__("pathlib").Path(config.__file__),
        __import__("pathlib").Path(mutations.__file__),
    ))
    assert "Aastha" not in sources
    assert "Praveen" not in sources
    assert "if user_id ==" not in sources


def test_configured_custom_schema_field_updates_dynamically(slack, monkeypatch):
    custom_schema = deepcopy(SCHEMA)
    custom_schema["schema"].append({"id": "estimate_col", "key": "estimate",
                                     "name": "Estimate", "type": "number"})
    item = slack.add("Alpha")
    item["fields"].append({"column_id": "estimate_col", "number": 3})
    monkeypatch.setattr(slack_tools, "get_list_schema", lambda list_id: custom_schema)
    monkeypatch.setitem(config.FIELD_CONTROLS, "L:estimate",
                        {"read": "view", "edit": "edit_estimate"})
    monkeypatch.setitem(config.PERMISSIONS, "admin", config.PERMISSIONS["admin"] | {"edit_estimate"})
    before = {field["column_id"]: deepcopy(field) for field in item["fields"]}
    ctx = main.context("UA", "C", "T", None, "W")
    result = main.handle_mutation({"intent": "update", "task_name": "Alpha", "tasks": [],
                                   "changes": [{"field": "estimate", "value": 8}]}, ctx, "unused")
    assert "updated" in result
    assert slack_tools.extract_field_value(item, custom_schema, "estimate") == 8
    after = {field["column_id"]: field for field in item["fields"]}
    for column_id in {"name", "done", "priority"}:
        assert after[column_id] == before[column_id]


@pytest.mark.parametrize("wording,metric", [
    ("Give me an overview of my progress", "overview"),
    ("How many of our action items are complete?", "completion"),
    ("Display the workload for Morgan", "workload"),
    ("Provide a breakdown by priority", "priority_distribution"),
    ("Is anything currently at risk?", "at_risk"),
])
def test_progress_language_normalizes_to_validated_metrics(wording, metric):
    parsed = commands.validate_command(intent_parser.parse_intent(wording))
    assert parsed["intent"] == "progress"
    assert metric in parsed["analytics_metrics"]


def test_progress_overview_uses_real_authorized_slack_items(slack):
    today = main.current_date()
    slack.add("Mine completed", completed=True, assignee="UA", priority="P1")
    slack.add("Mine overdue", assignee="UA", priority="P2", due=(today - timedelta(days=1)).isoformat())
    slack.add("Mine today", assignee="UA", priority="P3", due=today.isoformat())
    slack.add("Other task", assignee="UM", priority="P1")
    response = ask("Show my progress", thread="PROGRESS_SELF")
    assert "Total: 3" in response
    assert "Completed: 1" in response
    assert "Pending: 2" in response
    assert "Overdue: 1" in response
    assert "33.3%" in response
    assert "█" in response
    assert not slack.writes


def test_team_workload_visualization_uses_dynamic_member_names(slack):
    slack.add("One", assignee="UM")
    slack.add("Two", assignee="UM")
    slack.add("Three", assignee="UP")
    response = ask("Show the team's workload", thread="TEAM_WORKLOAD")
    assert "Pending workload by assignee" in response
    assert "Morgan" in response and "Praveen" in response
    assert "2" in response and "1" in response
    assert "█" in response
    assert not slack.writes


def test_member_cannot_view_another_members_progress(slack):
    slack.add("Private work", assignee="UP")
    response = ask("How is Praveen doing?", user="UM", thread="PROGRESS_RBAC")
    assert "Permission denied" in response
    assert not slack.writes


def test_completion_period_never_uses_due_date_as_completion_time(slack):
    slack.add("Completed without timestamp", completed=True, assignee="UA", due=main.current_date().isoformat())
    response = ask("What did we complete this week?", thread="PROGRESS_TIME_LIMIT")
    assert "No reliable timestamped records" in response
    assert "does not expose reliable completion timestamps" in response
    assert not slack.writes


def test_completion_time_series_uses_real_completion_timestamps(slack):
    today = main.current_date()
    first = slack.add("Timestamped one", completed=True)
    second = slack.add("Timestamped two", completed=True)
    outside = slack.add("Old timestamp", completed=True)
    first["completed_at"] = today.isoformat() + "T08:00:00Z"
    second["completed_at"] = today.isoformat() + "T12:00:00Z"
    outside["completed_at"] = (today - timedelta(days=14)).isoformat() + "T12:00:00Z"
    response = ask("What did we complete this week?", thread="PROGRESS_TIME")
    assert f"{today.isoformat()}:" in response
    assert " 2" in response
    assert (today - timedelta(days=14)).isoformat() not in response


def test_progress_task_result_becomes_exact_context_for_followup(slack):
    today = main.current_date()
    overdue = slack.add("Overdue exact", assignee="UA", due=(today - timedelta(days=1)).isoformat())
    slack.add("Future", assignee="UA", due=(today + timedelta(days=2)).isoformat())
    response = ask("Are there any overdue tasks?", thread="PROGRESS_CONTEXT")
    assert "Overdue exact" in response and "Future" not in response
    response = ask("complete the first one", thread="PROGRESS_CONTEXT")
    assert "Action item completed" in response
    assert slack_tools.extract_completed(overdue, SCHEMA)


def test_progress_priority_filter_and_distribution_are_composable(slack):
    slack.add("Critical one", priority="P1")
    slack.add("Critical two", priority="P1")
    slack.add("Normal", priority="P3")
    response = ask("Show the P1 workload and priority breakdown", thread="PROGRESS_COMPOSE")
    assert "Pending workload by assignee" in response
    assert "Priority distribution" in response
    assert "P1" in response and "P3" not in response


def test_progress_engine_reports_missing_dynamic_fields_without_guessing():
    schema = {"schema": [{"id": "name", "key": "name", "type": "text"}]}
    item = {"id": "I1", "fields": [{"column_id": "name", "text": "Only a name"}]}
    report = progress_engine.calculate_progress(
        [item], schema, today=main.current_date(), available_fields=set(),
        metrics=["overview", "at_risk"])
    rendered = progress_engine.render_progress(report, lambda items, title: title)
    assert "Completion: Unavailable" in rendered
    assert "At-risk tasks require" in rendered


def test_progress_distributions_and_deadlines_are_calculated_from_current_items(slack):
    today = main.current_date()
    slack.add("Done", completed=True, priority="P1", due=(today - timedelta(days=4)).isoformat())
    slack.add("Late", priority="P1", due=(today - timedelta(days=1)).isoformat())
    slack.add("Today", priority="P2", due=today.isoformat())
    slack.add("Later", priority="P3", due=(today + timedelta(days=1)).isoformat())
    report = progress_engine.calculate_progress(
        slack.items, SCHEMA, today=today,
        metrics=["overview", "status_distribution", "priority_distribution", "due_today", "due_this_week"])
    assert report.snapshot == {
        "total": 4, "completed": 1, "pending": 3, "overdue": 1,
        "due_today": 1, "due_this_week": 2, "completion_rate": 25.0,
    }
    assert report.status_distribution == {"Completed": 1, "Pending": 2, "Overdue": 1}
    assert report.priority_distribution == {"P1": 2, "P2": 1, "P3": 1}
    assert [item["id"] for item in report.due_today_items] == [slack.items[2]["id"]]


def test_created_time_series_uses_only_real_creation_metadata(slack):
    today = main.current_date()
    first = slack.add("Created one")
    second = slack.add("Created two")
    missing = slack.add("No creation date")
    first["created_at"] = today.isoformat() + "T08:00:00Z"
    second["created_timestamp"] = today.isoformat() + "T10:00:00Z"
    series = progress_engine.calculate_time_series(
        [first, second, missing], SCHEMA, "created",
        {"start": today.isoformat(), "end": today.isoformat()})
    assert series == {"values": {today.isoformat(): 2}, "available": 2, "missing": 1}


def test_progress_and_mutation_remain_independent_compound_operations():
    parsed = commands.validate_command(intent_parser.parse_intent(
        "Show my progress and then complete the release checklist"))
    assert parsed["intent"] == "compound"
    assert [operation["intent"] for operation in parsed["operations"]] == ["progress", "complete"]


def test_progress_uses_channel_specific_list_mapping(slack, monkeypatch):
    monkeypatch.setattr(config, "CHANNEL_LISTS", {"C": "LIST_A", "C2": "LIST_B"})
    ask("Show my progress", channel="C", thread="PROGRESS_CHANNEL_A")
    ask("Show my progress", channel="C2", thread="PROGRESS_CHANNEL_B")
    assert "LIST_A" in slack.list_requests
    assert "LIST_B" in slack.list_requests


def test_manager_can_view_another_members_progress(slack, monkeypatch):
    monkeypatch.setattr(config, "USER_ROLES", {"UM": "manager"})
    slack.add("Praveen task", assignee="UP")
    response = ask("How is Praveen doing?", user="UM", thread="PROGRESS_MANAGER")
    assert "Total: 1" in response
    assert "Permission denied" not in response
    assert not slack.writes


def test_progress_respects_dynamic_assignee_field_visibility(slack, monkeypatch):
    slack.add("Owned", assignee="UP")
    monkeypatch.setitem(config.FIELD_CONTROLS, "L:assignee", {"read": "private_assignee", "edit": "edit_assignee"})
    response = ask("Show the team's workload", thread="PROGRESS_FIELD_RBAC")
    assert "No reliable data available" in response
    assert "Assignee workload is unavailable or not readable" in response
    assert "Praveen" not in response
    assert not slack.writes


def test_untrusted_progress_metric_is_rejected():
    with pytest.raises(ValueError, match="supported progress metrics"):
        commands.validate_command({"intent": "progress", "analytics_metrics": ["invented_metric"]})


@pytest.mark.parametrize("wording,intent", [
    ("Which action items need attention?", "health"),
    ("Help me organize my work for this week", "plan"),
    ("Who on the team is overloaded?", "workload"),
    ("Prepare my daily standup", "standup"),
    ("Which action items are blocked?", "dependencies"),
    ("Who changed this task?", "history"),
])
def test_project_intelligence_language_maps_to_validated_intents(wording, intent):
    parsed = commands.validate_command(intent_parser.parse_intent(wording))
    assert parsed["intent"] == intent


def test_health_engine_reads_real_list_and_preserves_exact_context(slack):
    today = main.current_date()
    late = slack.add("Late delivery", assignee="UA", priority="P1",
                     due=(today - timedelta(days=2)).isoformat())
    slack.add("Safe delivery", assignee="UA", priority="P3",
              due=(today + timedelta(days=10)).isoformat())
    response = ask("Which tasks need attention?", thread="HEALTH")
    assert "Late delivery" in response and "Overdue by 2 days" in response
    assert "Safe delivery" not in response
    assert not slack.writes
    response = ask("complete the first one", thread="HEALTH")
    assert "Action item completed" in response
    assert slack_tools.extract_completed(late, SCHEMA)


def test_plan_is_proposal_only_and_keeps_exact_ids(slack):
    today = main.current_date()
    first = slack.add("Urgent plan item", assignee="UA", priority="P1",
                      due=(today + timedelta(days=1)).isoformat())
    second = slack.add("Later plan item", assignee="UA", priority="P3",
                       due=(today + timedelta(days=5)).isoformat())
    response = ask("Help me plan my tasks for this week", thread="PLAN")
    assert "Suggested Plan" in response
    assert "No Slack List fields were changed" in response
    assert not slack.writes
    proposal = main._state(main.context("UA", "C", "PLAN", None, "W"))["proposal"]
    assert [entry["item_id"] for entry in proposal["entries"]] == [first["id"], second["id"]]


def test_workload_recommendation_is_read_only_and_exact(slack):
    for index in range(6):
        slack.add(f"Heavy {index}", assignee="UM", priority="P1" if index < 2 else "P3")
    slack.add("Light", assignee="UP", priority="P3")
    response = ask("Suggest a better team workload balance", thread="BALANCE")
    assert "Team Workload" in response
    assert "Proposed balance" in response
    assert "No assignments were changed" in response
    assert not slack.writes
    proposal = main._state(main.context("UA", "C", "BALANCE", None, "W"))["proposal"]
    assert proposal["kind"] == "workload_balance"
    assert all(entry["item_id"].startswith("I") for entry in proposal["entries"])


def test_member_cannot_generate_cross_member_balance(slack):
    slack.add("Other", assignee="UP")
    response = ask("Suggest a better team workload balance", user="UM", thread="BALANCE_RBAC")
    assert "Permission denied" in response
    assert not slack.writes


def test_standup_does_not_invent_completed_today(slack):
    slack.add("Finished sometime", completed=True, assignee="UA")
    slack.add("Open now", assignee="UA")
    response = ask("Give me my standup", thread="STANDUP")
    assert "Completed (current List state)" in response
    assert "not as completed today" in response
    assert "Finished sometime" in response and "Open now" in response
    assert not slack.writes


def test_dependencies_fail_truthfully_when_schema_has_no_dependency_field(slack):
    slack.add("Task without dependency metadata")
    response = ask("Which tasks are blocked?", thread="DEPENDENCIES")
    assert "no explicit dependency or blocker field" in response
    assert "No dependency data was inferred" in response
    assert not slack.writes


@pytest.mark.parametrize("wording,expected", [
    ("Find Praveen's overdue P1 action items", {"assignees": ["Praveen"], "overdue": True, "priority": "P1"}),
    ("Show my work due this week", {"assignee_self": True, "due_this_week": True}),
    ("Find API-related tasks assigned to someone else", {"query": "API", "assignee_condition": "other"}),
    ("Find all high-priority incomplete tasks", {"priority": "P1", "completed": False}),
])
def test_advanced_search_concepts_compose_in_structured_query(wording, expected):
    parsed = commands.validate_command(intent_parser.parse_intent(wording))
    assert parsed["intent"] == "list"
    for key, value in expected.items():
        assert parsed[key] == value


def test_due_tomorrow_and_created_this_month_are_distinct_temporal_dimensions():
    tomorrow = (main.current_date() + timedelta(days=1)).isoformat()
    due = commands.validate_command(intent_parser.parse_intent("Which tasks are due tomorrow?"))
    created = commands.validate_command(intent_parser.parse_intent("Show pending tasks created this month"))
    assert due["temporal_filter"] == {"field": "due_date", "relation": "on", "date": tomorrow}
    assert created["temporal_filter"]["field"] == "created_at"
    assert created["temporal_filter"]["relation"] == "between"


def test_search_applies_text_relative_assignee_priority_and_status_together(slack):
    today = main.current_date()
    wanted = slack.add("API gateway", assignee="UP", priority="P1",
                       due=(today + timedelta(days=1)).isoformat())
    slack.add("API docs", assignee="UA", priority="P1",
              due=(today + timedelta(days=1)).isoformat())
    slack.add("UI gateway", assignee="UP", priority="P1",
              due=(today + timedelta(days=1)).isoformat())
    response = ask("Find pending P1 API-related tasks assigned to someone else", thread="ADV_SEARCH")
    assert "API gateway" in response
    assert "API docs" not in response and "UI gateway" not in response
    state = main._state(main.context("UA", "C", "ADV_SEARCH", None, "W"))
    assert [entry["item_id"] for entry in state["displayed_tasks"]] == [wanted["id"]]
    assert not slack.writes


def test_completed_temporal_search_uses_real_timestamp_when_available(slack):
    today = main.current_date()
    item = slack.add("Completed today", completed=True, assignee="UA")
    item["completed_at"] = today.isoformat() + "T09:00:00Z"
    slack.add("Completed without timestamp", completed=True, assignee="UA")
    response = ask("What tasks were completed today?", thread="COMPLETED_TIME")
    assert "Completed today" in response
    assert "Completed without timestamp" not in response


def test_likely_duplicate_requires_scoped_confirmation_before_create(slack):
    slack.add("Prepare client report", assignee="UA")
    response = ask("create Prepare client reports for me", thread="DUP_CONFIRM")
    assert "Possible Duplicate" in response
    assert len(slack.items) == 1
    assert not slack.writes
    response = ask("confirm", thread="DUP_CONFIRM")
    assert "created successfully" in response
    assert len(slack.items) == 2


def test_duplicate_confirmation_is_invalidated_by_external_change(slack):
    existing = slack.add("Prepare client report", assignee="UA")
    assert "Possible Duplicate" in ask("create Prepare client reports for me", thread="DUP_STALE")
    existing["fields"].append({"column_id": "due", "date": [(main.current_date() + timedelta(days=3)).isoformat()]})
    response = ask("confirm", thread="DUP_STALE")
    assert "changed after confirmation was requested" in response
    assert len(slack.items) == 1
    assert not slack.writes


def test_large_destructive_collection_requires_fresh_confirmation(slack):
    for index in range(config.CONFIRMATION_THRESHOLD):
        slack.add(f"Delete target {index}", assignee="UA")
    response = ask("delete all action items", thread="BULK_CONFIRM")
    assert "Confirmation Required" in response
    assert len(slack.items) == config.CONFIRMATION_THRESHOLD
    assert not slack.writes
    response = ask("confirm", thread="BULK_CONFIRM")
    assert "Action items deleted" in response
    assert not slack.items
    assert len(slack.writes) == config.CONFIRMATION_THRESHOLD


def test_stale_bulk_confirmation_never_mutates(slack):
    items = [slack.add(f"Bulk {index}", assignee="UA") for index in range(config.CONFIRMATION_THRESHOLD)]
    assert "Confirmation Required" in ask("complete all action items", thread="BULK_STALE")
    items[0]["fields"].append({"column_id": "due", "date": [(main.current_date() + timedelta(days=2)).isoformat()]})
    response = ask("confirm", thread="BULK_STALE")
    assert "changed after confirmation was requested" in response
    assert not slack.writes
    assert all(not slack_tools.extract_completed(item, SCHEMA) for item in items)


def test_cancel_clears_confirmation_without_mutation(slack):
    for index in range(config.CONFIRMATION_THRESHOLD):
        slack.add(f"Cancel {index}", assignee="UA")
    ask("delete all action items", thread="BULK_CANCEL")
    response = ask("cancel", thread="BULK_CANCEL")
    assert "Cancelled" in response
    assert not slack.writes
    assert "no current confirmation" in ask("confirm", thread="BULK_CANCEL")


def test_apply_plan_refetches_authorizes_updates_and_verifies(slack):
    today = main.current_date()
    first = slack.add("Plan one", assignee="UA", priority="P1", due=(today + timedelta(days=3)).isoformat())
    second = slack.add("Plan two", assignee="UA", priority="P2", due=(today + timedelta(days=4)).isoformat())
    ask("Plan my work for this week", thread="PLAN_APPLY")
    response = ask("apply this plan", thread="PLAN_APPLY")
    assert "Proposal applied and verified" in response
    assert slack_tools.extract_item_id(first) in {entry["id"] for entry in slack.items}
    assert slack_tools.extract_item_id(second) in {entry["id"] for entry in slack.items}
    assert len(slack.writes) == 2


def test_plan_application_fails_closed_when_target_changes(slack):
    today = main.current_date()
    item = slack.add("Plan stale", assignee="UA", priority="P1", due=(today + timedelta(days=3)).isoformat())
    ask("Plan my work for this week", thread="PLAN_STALE")
    item["fields"].append({"column_id": "external", "text": "changed"})
    response = ask("apply this plan", thread="PLAN_STALE")
    assert "changed or no longer exist" in response
    assert not slack.writes


def test_verified_mutation_is_available_in_audit_history(slack):
    item = slack.add("Audited", assignee="UA")
    ask("complete Audited", thread="AUDIT")
    response = ask("Who changed this task?", thread="AUDIT")
    assert "Verified mutation history" in response
    assert "complete" in response
    assert item["id"] in response


def test_week_over_week_progress_uses_real_completion_timestamps(slack):
    today = main.current_date()
    current_start = today - timedelta(days=today.weekday())
    previous_start = current_start - timedelta(days=7)
    current = slack.add("Current completion", completed=True)
    previous_a = slack.add("Previous completion A", completed=True)
    previous_b = slack.add("Previous completion B", completed=True)
    current["completed_at"] = current_start.isoformat() + "T09:00:00Z"
    previous_a["completed_at"] = previous_start.isoformat() + "T09:00:00Z"
    previous_b["completed_at"] = (previous_start + timedelta(days=1)).isoformat() + "T09:00:00Z"
    parsed = commands.validate_command(intent_parser.parse_intent("Compare this week with last week"))
    assert parsed["intent"] == "progress"
    assert parsed["analytics_metrics"] == ["comparison"]
    response = ask("Compare this week with last week", thread="PROGRESS_COMPARE")
    assert "Completed-task comparison" in response
    assert "This period" in response and "Previous period" in response
    assert not slack.writes


def test_health_explanation_resolves_named_and_contextual_exact_targets(slack):
    today = main.current_date()
    target = slack.add("Release gate", assignee="UA", priority="P1",
                       due=(today + timedelta(days=1)).isoformat())
    slack.add("Other item", assignee="UA", priority="P3",
              due=(today + timedelta(days=8)).isoformat())
    named = ask("Why is Release gate marked as needing attention?", thread="HEALTH_EXPLAIN")
    assert "Release gate" in named and "Due tomorrow" in named and "P1 priority" in named
    ask("show my tasks", thread="HEALTH_CONTEXT")
    contextual = ask("What's the health of those tasks?", thread="HEALTH_CONTEXT")
    assert "Release gate" in contextual and "Other item" in contextual
    state = main._state(main.context("UA", "C", "HEALTH_CONTEXT", None, "W"))
    assert target["id"] in [entry["item_id"] for entry in state["displayed_tasks"]]
    assert not slack.writes


def test_explicit_dependency_field_is_discovered_and_reported(slack, monkeypatch):
    schema = deepcopy(SCHEMA)
    schema["schema"].append({"id": "depends", "key": "depends_on", "name": "Depends On", "type": "text"})
    item = slack.add("Deploy service", assignee="UA")
    item["fields"].append({"column_id": "depends", "text": "Approve release"})
    monkeypatch.setattr(slack_tools, "get_list_schema", lambda list_id: schema)
    monkeypatch.setitem(config.FIELD_CONTROLS, "L:depends_on", {"read": "view", "edit": "edit_name"})
    response = ask("Which tasks are blocked?", thread="DEPENDENCY_DATA")
    assert "Deploy service" in response
    assert "Approve release" in response
    assert not slack.writes


def test_workload_proposal_applies_exact_existing_item(slack):
    for index in range(6):
        slack.add(f"Owned heavy {index}", assignee="UM", priority="P1" if index < 2 else "P3")
    slack.add("Owned light", assignee="UP", priority="P3")
    ask("Recommend a better team workload balance", thread="BALANCE_APPLY")
    proposal = main._state(main.context("UA", "C", "BALANCE_APPLY", None, "W"))["proposal"]
    target_id = proposal["entries"][0]["item_id"]
    destination = proposal["entries"][0]["changes"][0]["value"]
    response = ask("apply the proposal", thread="BALANCE_APPLY")
    assert "Proposal applied and verified" in response
    target = next(item for item in slack.items if item["id"] == target_id)
    assert slack_tools.extract_assignee_ids(target, SCHEMA) == [destination]


def test_confirmation_is_time_limited_and_conversation_scoped(slack):
    for index in range(config.CONFIRMATION_THRESHOLD):
        slack.add(f"Scoped {index}", assignee="UA")
    ask("delete all action items", thread="CONFIRM_HOME")
    assert "no current confirmation" in ask("confirm", thread="CONFIRM_OTHER")
    ctx = main.context("UA", "C", "CONFIRM_HOME", None, "W")
    state = main._state(ctx)
    state["confirmation"]["expires_at"] = 0
    main._save_state(ctx, state)
    assert "expired" in ask("confirm", thread="CONFIRM_HOME")
    assert not slack.writes


def test_visualization_renderer_consumes_structured_report_only():
    import visualization
    report = progress_engine.ProgressReport(
        requested=("priority_distribution",),
        priority_distribution={"P1": 2, "P2": 1})
    rendered = visualization.render_progress(report, lambda items, title: title)
    assert "Priority distribution" in rendered
    assert "P1" in rendered and "█" in rendered


def test_dependency_graph_answers_impact_only_from_explicit_values(slack, monkeypatch):
    schema = deepcopy(SCHEMA)
    schema["schema"].append({"id": "depends", "key": "depends_on", "name": "Depends On", "type": "text"})
    origin = slack.add("Approve release", assignee="UA")
    dependent = slack.add("Deploy service", assignee="UA")
    dependent["fields"].append({"column_id": "depends", "text": origin["id"]})
    monkeypatch.setattr(slack_tools, "get_list_schema", lambda list_id: schema)
    monkeypatch.setitem(config.FIELD_CONTROLS, "L:depends_on", {"read": "view", "edit": "edit_name"})
    response = ask("What becomes available if Approve release is completed?", thread="DEPENDENCY_IMPACT")
    assert "Deploy service" in response
    assert "satisfy one explicit dependency" in response
    assert not slack.writes
