import importlib
import sys
from types import SimpleNamespace

import pytest
from src import graph as intent_parser

from src.config import RuntimeConfig, is_lambda_runtime, state_db_path


@pytest.mark.parametrize("utterance,owner,status,temporal_field", [
    ("what am I responsible for?", "self", "open", None),
    ("what does Praveen still have open?", "Praveen", "open", None),
    ("can you show me Praveen’s outstanding work?", "Praveen", "open", None),
    ("what needs to be done today?", None, "open", "due_date"),
    ("what did I finish today?", "self", "completed", "completed_at"),
    ("what is due before Friday?", None, "open", "due_date"),
])
def test_semantic_work_reads_are_structured_without_ollama(
        monkeypatch, utterance, owner, status, temporal_field):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("Ollama called"))
    parsed = intent_parser.parse_intent(utterance)
    assert parsed["intent"] == "list"
    assert parsed["status"] == status
    if owner == "self":
        assert parsed.get("assignee_self") is True
    elif owner:
        assert owner in parsed.get("assignees", [])
    if temporal_field:
        assert parsed["temporal_filter"]["field"] == temporal_field


@pytest.mark.parametrize("utterance,title,assignee,priority,due", [
    ("create a task called deployment verification, assign it to me, priority P2, due tomorrow",
     "deployment verification", None, "P2", "2026-10-08"),
    ("please add deployment verification to my tasks",
     "deployment verification", None, None, None),
    ("Could you please make sure Praveen handles the client report by Friday and mark it high priority?",
     "Handle the client report", "Praveen", "P1", "2026-10-09"),
])
def test_create_instruction_metadata_never_enters_title(
        monkeypatch, utterance, title, assignee, priority, due):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("Ollama called"))
    parsed = intent_parser.parse_intent(utterance)
    assert parsed["intent"] == "create"
    assert parsed["task_name"] == title
    assert parsed.get("assignee") == assignee
    assert parsed.get("priority") == priority
    assert parsed.get("due_date") == due
    assert not any(word in title.casefold() for word in ("assign it", "priority p", "due tomorrow"))


def test_one_message_has_same_request_key_across_slack_event_subscriptions():
    from src import app
    mention = {"event_id": "Ev-mention", "team_id": "T1"}
    message = {"event_id": "Ev-message", "team_id": "T1"}
    shared = {"channel": "C1", "user": "U1", "ts": "1728.010"}
    assert app.request_key(mention, {**shared, "type": "app_mention", "client_msg_id": "client-1"}, "list my tasks") == (
        app.request_key(message, {**shared, "type": "message"}, "list my tasks"))


@pytest.mark.parametrize("utterance,expected_date", [
    ("Move the deployment report to Praveen, make it P1, and push the deadline to Monday.",
     "2026-10-12"),
    ("Give the client report to Praveen and make it P1 for Friday.",
     "2026-10-09"),
])
def test_compound_assignment_fields_update_one_task_without_ollama(
        monkeypatch, utterance, expected_date):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("Ollama called"))
    parsed = intent_parser.parse_intent(utterance)
    assert parsed["intent"] == "update"
    assert parsed["task_name"].casefold().endswith("report")
    assert {change["field"]: change["value"] for change in parsed["changes"]} == {
        "assignee": "Praveen", "priority": "P1", "due_date": expected_date}


def test_state_database_path_is_local_by_default_and_writable_in_lambda():
    assert state_db_path({}) == "slack_list_state.sqlite3"
    assert state_db_path({"STATE_DB": "slack_list_state.sqlite3"}) == "slack_list_state.sqlite3"
    assert state_db_path({"AWS_LAMBDA_FUNCTION_NAME": "slack-list-assistant"}) == "/tmp/slack_list_state.sqlite3"
    assert state_db_path({"AWS_LAMBDA_FUNCTION_NAME": "slack-list-assistant",
                          "STATE_DB": "slack_list_state.sqlite3"}) == "/tmp/slack_list_state.sqlite3"
    assert state_db_path({"AWS_LAMBDA_FUNCTION_NAME": "slack-list-assistant",
                          "STATE_DB": "/tmp/custom.sqlite3"}) == "/tmp/custom.sqlite3"


@pytest.mark.parametrize("phrase,owner,completed", [
    ("list my task", "self", False), ("list my tasks", "self", False),
    ("show my tasks", "self", False), ("list all task", None, None),
    ("list all tasks", None, None),
    ("ist all @Praveen tasks", "@Praveen", False),
    ("list all @Praveen tasks", "@Praveen", False),
    ("list @Praveen tasks", "@Praveen", False),
    ("show @Praveen tasks", "@Praveen", False),
    ("show Praveen's tasks", "Praveen", False),
    ("list Praveen tasks", "Praveen", False),
    ("list all <@U0C1YNSR82W> tasks", "<@U0C1YNSR82W>", False),
    ("list completed @Praveen tasks", "@Praveen", True),
    ("list pending @Praveen tasks", "@Praveen", False),
    ("completed tasks", None, True), ("pending tasks", None, False),
])
def test_task_listing_is_deterministic_without_ollama(monkeypatch, phrase, owner, completed):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("Ollama called"))
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == "list"
    assert parsed.get("completed") is completed
    if owner == "self":
        assert parsed["assignee_self"] is True
    elif owner:
        assert parsed.get("assignee") == owner or owner in parsed.get("assignees", [])


@pytest.mark.parametrize("phrase,expected_intent,expected_title,assignee,due", [
    ("Prepare project report, assign to me", "create", "Prepare project report", None, None),
    ("Prepare project report to @AasthaA", "create", "Prepare project report", "AasthaA", None),
    ("assign client report to @Praveen", "update", "client report", None, None),
    ("docs task till this friday", "create", "docs task", None, "2026-10-09"),
])
def test_title_extraction_removes_assignment_and_date_syntax(
        monkeypatch, phrase, expected_intent, expected_title, assignee, due):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("Ollama called"))
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == expected_intent
    assert parsed["task_name"].casefold() == expected_title.casefold()
    if assignee:
        assert parsed["assignee"] == assignee
    if due:
        assert parsed["due_date"] == due
    assert not any(fragment in parsed["task_name"].casefold()
                   for fragment in ("assign to", "till this", "to @", "priority p"))


@pytest.mark.parametrize("phrase", [
    "please make sure Aastha reviews the deployment checklist before October 10",
    "make sure Aastha reviews the deployment checklist by October 10",
    "Aastha should review the deployment checklist before October 10",
    "please have Aastha review the deployment checklist before October 10",
    "assign Aastha to review the deployment checklist before October 10",
    "Aastha needs to review the deployment checklist by October 10",
])
def test_delegated_create_extracts_action_person_and_date_without_llm(monkeypatch, phrase):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("LLM called"))
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == "create"
    assert parsed["task_name"] == "Review the deployment checklist"
    assert parsed["assignee"] == "Aastha"
    assert parsed["due_date"] == "2026-10-10"


@pytest.mark.parametrize("phrase", [
    "we need to finish the client report and assign it to Praveen",
    "finish the client report and give it to Praveen",
    "have Praveen finish the client report",
    "Praveen should finish the client report",
])
def test_delegated_create_strips_assignment_instruction(monkeypatch, phrase):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("LLM called"))
    parsed = intent_parser.parse_intent(phrase)
    assert (parsed["intent"], parsed["task_name"], parsed["assignee"]) == (
        "create", "Finish the client report", "Praveen")


def test_delegated_create_keeps_mention_and_explicit_quoted_title():
    parsed = intent_parser.parse_intent(
        "please make sure @AasthaA reviews the deployment checklist before October 10")
    assert parsed["task_name"] == "Review the deployment checklist"
    assert parsed["assignee"] == "@AasthaA"
    quoted = intent_parser.parse_intent('create a task called "Reviewing the deployment process"')
    assert quoted["task_name"] == "Reviewing the deployment process"


def test_graph_exposes_existing_pipeline_and_deterministic_parser(monkeypatch):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("LLM called"))
    parsed = intent_parser.parse_intent("show control tower")
    assert parsed["intent"] == "control_tower"
    assert intent_parser.validate_command(parsed)["intent"] == "control_tower"


def test_lambda_configuration_does_not_require_socket_mode_token(lambda_environment):
    assert is_lambda_runtime(lambda_environment)
    settings = RuntimeConfig.from_environment()
    assert settings.app_token is None
    settings.validate(socket_mode=False)


def test_lambda_handler_imports_without_app_token(lambda_environment):
    module = importlib.import_module("src.handler")
    assert callable(module.lambda_handler)


def test_lambda_handler_uses_shared_app_without_socket_token(lambda_environment, monkeypatch):
    module = importlib.import_module("src.handler")
    module._request_handler = None
    shared_app = object()
    handled = []

    class FakeRequestHandler:
        def __init__(self, app):
            assert app is shared_app

        def handle(self, event, context):
            handled.append((event, context))
            return {"statusCode": 200}

    monkeypatch.setitem(
        sys.modules, "slack_bolt.adapter.aws_lambda",
        SimpleNamespace(SlackRequestHandler=FakeRequestHandler))
    monkeypatch.setattr(module, "get_app", lambda: shared_app)
    assert module.lambda_handler({"body": "test"}, None) == {"statusCode": 200}
    assert handled == [({"body": "test"}, None)]


def _lambda_bolt_request(monkeypatch):
    """Exercise the installed Bolt AWS adapter without network access."""
    import hashlib
    import hmac
    import json
    import time
    from slack_sdk import WebClient
    from slack_sdk.web.slack_response import SlackResponse
    from unittest.mock import Mock
    from src import app as shared_app
    from src import handler
    from src import slack_client

    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "slack-list-assistant-test")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *args, **kwargs: None))
    monkeypatch.setattr(shared_app, "SIGNING_SECRET", "test-signing-secret")
    monkeypatch.setattr(slack_client, "configure", lambda client: None)
    client = WebClient(token="xoxb-test")
    client.auth_test = Mock(return_value=SlackResponse(
        client=client, http_verb="POST", api_url="", req_args={},
        data={"ok": True, "user_id": "UBOT", "bot_id": "BBOT", "team_id": "T"},
        headers={}, status_code=200))
    bolt = shared_app.create_app(client=client)
    monkeypatch.setattr(handler, "get_app", lambda: bolt)
    monkeypatch.setattr(handler, "_request_handler", None)

    def send(payload, *, valid=True, encoded=False, content_type="application/json"):
        import base64
        body = json.dumps(payload) if isinstance(payload, dict) else payload
        timestamp = str(int(time.time()))
        signature = "v0=" + hmac.new(b"test-signing-secret",
                                     f"v0:{timestamp}:{body}".encode(), hashlib.sha256).hexdigest()
        event = {
            "version": "2.0", "routeKey": "$default", "rawPath": "/",
            "requestContext": {"http": {"method": "POST", "path": "/"}},
            "headers": {"content-type": content_type,
                        "x-slack-request-timestamp": timestamp},
            "body": base64.b64encode(body.encode()).decode() if encoded else body,
            "isBase64Encoded": encoded,
        }
        if valid:
            event["headers"]["x-slack-signature"] = signature if valid is True else "v0=" + "0" * 64
        return handler.lambda_handler(event, SimpleNamespace(
            function_name="slack-list-assistant-test", invoked_function_arn="arn:aws:lambda:test"))

    return shared_app, bolt, send


def test_lambda_function_url_verification_and_signature(monkeypatch, caplog):
    from src import app as shared_app
    original = shared_app.app
    try:
        shared_app, bolt, send = _lambda_bolt_request(monkeypatch)
        assert bolt.process_before_response is True
        with caplog.at_level("INFO"):
            response = send({"type": "url_verification", "challenge": "challenge-test"}, encoded=True)
        assert response["statusCode"] == 200
        assert "challenge-test" in response["body"]
        assert "lambda_request_received" in caplog.text
        assert "bolt_request_handling_completed" in caplog.text
        assert "test-signing-secret" not in caplog.text
        assert send({"type": "url_verification", "challenge": "bad"}, valid=False)["statusCode"] == 401
        assert send({"type": "url_verification", "challenge": "bad"}, valid="bad")["statusCode"] == 401
    finally:
        shared_app.app = original


@pytest.mark.parametrize("request_text,expected_all", [
    ("list my tasks", False), ("list all tasks", True),
])
def test_lambda_app_mention_reaches_existing_list_parser(monkeypatch, caplog,
                                                         request_text, expected_all):
    from src import app as shared_app
    original = shared_app.app
    try:
        shared_app, bolt, send = _lambda_bolt_request(monkeypatch)
        delivered = []
        monkeypatch.setattr(shared_app, "_deliver", lambda *args, **kwargs: delivered.append(args))
        payload = {"type": "event_callback", "team_id": "T", "event_id": "Ev-1",
                   "event": {"type": "app_mention", "text": f"<@UBOT> {request_text}",
                             "user": "U123", "channel": "C123", "ts": "1.0"}}
        with caplog.at_level("INFO"):
            response = send(payload)
        assert response["statusCode"] == 200
        assert len(delivered) == 1 and delivered[0][1] == request_text
        parsed = intent_parser.parse_intent(delivered[0][1])
        assert parsed["intent"] == "list"
        assert bool(parsed.get("all_tasks")) is expected_all
        assert "listener_matched listener=app_mention" in caplog.text
        assert "lambda_response_returned status=200" in caplog.text
    finally:
        shared_app.app = original


def test_lambda_message_and_add_command_use_registered_listeners(monkeypatch):
    from src import app as shared_app
    original = shared_app.app
    try:
        shared_app, bolt, send = _lambda_bolt_request(monkeypatch)
        delivered = []
        monkeypatch.setattr(shared_app, "_deliver", lambda *args, **kwargs: delivered.append(args))
        message = {"type": "event_callback", "team_id": "T", "event_id": "Ev-2",
                   "event": {"type": "message", "text": "list my tasks", "user": "U123",
                             "channel": "D123", "ts": "2.0"}}
        assert send(message)["statusCode"] == 200
        assert delivered[0][1] == "list my tasks"
        command = "command=%2Fadd&text=Review+report&user_id=U123&channel_id=C123&team_id=T&trigger_id=TR1"
        assert send(command, content_type="application/x-www-form-urlencoded")["statusCode"] == 200
        assert delivered[1][1] == "add Review report"
    finally:
        shared_app.app = original


def test_local_bolt_keeps_default_listener_mode(monkeypatch):
    from slack_sdk import WebClient
    from slack_sdk.web.slack_response import SlackResponse
    from unittest.mock import Mock
    from src import app as shared_app
    from src import slack_client
    original = shared_app.app
    try:
        monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
        monkeypatch.delenv("AWS_EXECUTION_ENV", raising=False)
        monkeypatch.setattr(slack_client, "configure", lambda client: None)
        client = WebClient(token="xoxb-test")
        client.auth_test = Mock(return_value=SlackResponse(
            client=client, http_verb="POST", api_url="", req_args={},
            data={"ok": True, "user_id": "UBOT", "bot_id": "BBOT", "team_id": "T"},
            headers={}, status_code=200))
        bolt = shared_app.create_app(client=client)
        assert bolt.process_before_response is False
    finally:
        shared_app.app = original


def test_lambda_app_logs_info_even_when_runtime_configured_root_logger():
    import os
    import subprocess
    from pathlib import Path

    env = os.environ.copy()
    env.update(PYTHON_DOTENV_DISABLED="1", AWS_LAMBDA_FUNCTION_NAME="test", LOG_LEVEL="INFO")
    command = [sys.executable, "-c",
               "import logging; logging.basicConfig(level=logging.WARNING); "
               "import src.app; print(logging.getLogger('slack_list.lambda').getEffectiveLevel())"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "20"


# Migrated test coverage from test_completion.py
import pytest
from src.graph import parse_intent
from src import slack_client as slack_tools

class TestNaturalCompletionParsing:
    @pytest.mark.parametrize("phrase,expected_task", [
        ("I finished the login bug.", "login bug"),
        ("I finished login bug.", "login bug"),
        ("Login bug is finished.", "login bug"),
        ("I completed the login bug.", "login bug"),
        ("I have completed the login bug.", "login bug"),
        ("I did the login bug.", "login bug"),
        ("I have done the login bug.", "login bug"),
        ("I'm done with the login bug.", "login bug"),
        ("Login bug is done.", "login bug"),
        ("Mark login bug as completed.", "login bug"),
        ("Mark the login bug done.", "login bug"),
        ("I wrapped up the login bug.", "login bug"),
        ("The login bug is complete.", "login bug"),
        ("I've taken care of the login bug.", "login bug"),
        ("Done with the login bug.", "login bug"),
        ("Finished with login bug.", "login bug"),
        ("I already finished that task.", "__LAST__"),
        ("I already completed that.", "__LAST__"),
    ])
    def test_single_task_completion_phrases(self, phrase, expected_task):
        res = parse_intent(phrase)
        assert res.get("intent") == "complete", f"Failed intent for: {phrase}, got: {res}"
        task_name = (res.get("task_name") or "").lower()
        assert task_name == expected_task.lower(), f"Failed task_name for: {phrase}, got: {res.get('task_name')!r}"

    def test_multi_task_completion(self):
        text = "I finished the login bug and completed the dashboard deployment."
        res = parse_intent(text)
        assert res.get("intent") == "complete", f"Expected complete intent, got: {res}"
        tasks = res.get("tasks")
        assert isinstance(tasks, list), f"Expected tasks list, got: {tasks}"
        assert len(tasks) >= 2, f"Expected 2+ tasks, got: {len(tasks)}"
        names = [t.get("task_name", "").lower() for t in tasks]
        assert any("login bug" in n for n in names)
        assert any("dashboard" in n for n in names)


class TestSemanticMatching:
    def setup_method(self):
        self.mock_schema = {
            "schema": [
                {"id": "col_name", "key": "name", "name": "Task Name", "type": "text"},
                {"id": "col_assignee", "key": "assignee", "name": "Assignee", "type": "user"},
                {"id": "col_completed", "key": "completed", "name": "Completed", "type": "checkbox"},
            ]
        }

    def _make_item(self, id_val, name, completed=False, user_id=None):
        return {
            "id": id_val,
            "fields": [
                {"column_id": "col_name", "text": name},
                {"column_id": "col_completed", "checkbox": completed},
                {"column_id": "col_assignee", "user": [{"id": user_id}]} if user_id else {},
            ]
        }

    def test_exact_match(self):
        items = [self._make_item("1", "Login Bug")]
        matches = slack_tools.find_matches(items, "login bug", self.mock_schema)
        assert len(matches) == 1
        assert matches[0]["id"] == "1"

    def test_query_without_verb_matches_title_with_verb(self):
        items = [self._make_item("1", "Fix the login bug")]
        matches = slack_tools.find_matches(items, "login bug", self.mock_schema)
        assert len(matches) == 1
        assert matches[0]["id"] == "1"

    def test_query_with_verb_matches_title_without_verb(self):
        items = [self._make_item("1", "login bug")]
        matches = slack_tools.find_matches(items, "Fix the login bug", self.mock_schema)
        assert len(matches) == 1
        assert matches[0]["id"] == "1"

    def test_continuous_prefix_matching(self):
        items = [self._make_item("1", "Insight dashboard Completion for pitch")]
        matches = slack_tools.find_matches(items, "Insight dashboard", self.mock_schema)
        assert len(matches) == 1
        assert matches[0]["id"] == "1"

    def test_ambiguity_returns_multiple(self):
        items = [
            self._make_item("1", "Client report draft"),
            self._make_item("2", "Client report review"),
        ]
        matches = slack_tools.find_matches(items, "Client report", self.mock_schema)
        assert len(matches) == 2

    def test_false_positive_prevention(self):
        items = [self._make_item("1", "onboarding docs")]
        matches = slack_tools.find_matches(items, "docs check", self.mock_schema)
        assert len(matches) == 0


class TestListQueryScopes:
    @pytest.mark.parametrize("query", [
        "what are my tasks",
        "what do I need to work on?",
        "what is left for me?",
        "what should I do today?",
        "list my tasks",
        "show tasks",
    ])
    def test_pending_default_queries(self, query):
        res = parse_intent(query)
        assert res.get("intent") == "list"
        assert res.get("completed") is False or res.get("status") == "open" or "completed" not in res

    @pytest.mark.parametrize("query", [
        "what have I completed?",
        "what did I finish?",
        "show my completed tasks",
        "list completed items",
    ])
    def test_completed_queries(self, query):
        res = parse_intent(query)
        assert res.get("intent") == "list"
        assert res.get("completed") is True or res.get("status") == "completed"

    @pytest.mark.parametrize("query", [
        "show all my tasks",
        "show everything",
        "show all action items",
        "list all tasks",
    ])
    def test_all_tasks_queries(self, query):
        res = parse_intent(query)
        assert res.get("intent") == "list"
        assert res.get("all_tasks") is True or res.get("completed") is None



# Migrated test coverage from test_multi_create.py
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

from src.graph import parse_intent, _local_parse_create, _today, _tomorrow

# Silence noisy logs during test runs
logging.basicConfig(level=logging.CRITICAL)

_test_multi_create_TZ = ZoneInfo("Asia/Kathmandu")
_test_multi_create_TODAY = datetime.now(_test_multi_create_TZ).date()
TOMORROW = _test_multi_create_TODAY + timedelta(days=1)
_test_multi_create_TODAY_ISO = _test_multi_create_TODAY.isoformat()
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
            assert t.get("due_date") == _test_multi_create_TODAY_ISO, (
                f"Expected today={_test_multi_create_TODAY_ISO!r} for each task, got {t.get('due_date')!r}"
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
            assert t.get("due_date") == _test_multi_create_TODAY_ISO, f"Expected today, got {t.get('due_date')!r}"


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
                assert t.get("due_date") == _test_multi_create_TODAY_ISO, f"Expected today, got {t.get('due_date')!r}"
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
                assert t.get("due_date") == _test_multi_create_TODAY_ISO
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
                assert t.get("due_date") == _test_multi_create_TODAY_ISO, f"Expected {_test_multi_create_TODAY_ISO}, got {t.get('due_date')}"

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



# Migrated test coverage from test_parse.py
"""Manual integration probe; never executed during test collection."""

if __name__ == "__main__":
    from src.graph import parse_intent
    import json

    print("TEST 1:", json.dumps(parse_intent("assign New Task to @AasthaA due date 2026-09-17 and P1"), indent=2))



# Migrated test coverage from test_parser.py
"""Manual integration probe; never executed during test collection."""

if __name__ == "__main__":
    from src.graph import parse_intent
    import json

    tests = [
        "What is the weather?",
        "Add an action item to fix the login bug.",
        "Make task 123 P1.",
        "Show Aashu's tasks.",
        "Update task 123 to P2",
        "Set the priority of the login task to P1",
        "Mark task 123 as complete",
        "Remove task 123",
    ]

    for t in tests:
        print(f"Input: {t}")
        print(f"Output: {json.dumps(parse_intent(t), indent=2)}")
        print("-" * 40)



# Migrated test coverage from test_robustness.py
import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch

from src import config
from src import app as main
from src import slack_client as slack_tools
from src.graph import parse_intent, _local_parse, _local_parse_create, _local_parse_mutation

_test_robustness_TZ = ZoneInfo("Asia/Kathmandu")
_test_robustness_TODAY = datetime.now(_test_robustness_TZ).date()
_test_robustness_TODAY_ISO = _test_robustness_TODAY.isoformat()

_test_robustness_SCHEMA = {
    "schema": [
        {"id": "col_name", "key": "name", "name": "Task Name", "type": "text"},
        {"id": "col_assignee", "key": "assignee", "name": "Assignee", "type": "user"},
        {"id": "col_due", "key": "due_date", "name": "Due Date", "type": "date"},
        {"id": "col_priority", "key": "priority", "name": "Priority", "type": "select"},
        {"id": "col_completed", "key": "completed", "name": "Completed", "type": "checkbox"},
    ]
}

def _test_robustness_make_item(item_id, name, completed=False, user_id="U12345", due="2026-09-20", priority="P2"):
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
        ("low", "P3"),
        ("lowest", "P3"),
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
    def test_create_task_infinitive_is_title_not_assignee(self):
        res = parse_intent(
            "create a task to prepare the internship demo checklist for Praveen "
            "by October 8 with priority P2")
        assert res.get("intent") == "create"
        assert res.get("task_name") == "prepare the internship demo checklist"
        assert res.get("assignee") == "Praveen"
        assert res.get("due_date") == "2026-10-08"
        assert res.get("priority") == "P2"

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
        assert res.get("due_date") == (_test_robustness_TODAY + timedelta(days=1)).isoformat()


class TestCompletionVerification:
    def test_verified_completion(self):
        items_store = [_test_robustness_make_item("item_1", "Feature X", completed=False)]

        def mock_complete(item_id, ctx, list_id):
            items_store[0]["fields"][4]["checkbox"] = True
            items_store[0]["fields"][4]["value"] = True
            return True

        with patch.object(slack_tools, "get_list_schema", return_value=_test_robustness_SCHEMA),\
             patch.object(slack_tools, "list_action_items", side_effect=lambda ctx, list_id: items_store),\
             patch.object(slack_tools, "complete_action_item", side_effect=mock_complete),\
             patch.object(config, "has_permission", return_value=True):

            ctx = config.build_context(user_id="U12345", channel_id="C_DEV")
            parsed = {"intent": "complete", "task_name": "Feature X"}
            msg = main.handle_mutation(parsed, ctx, "test_mem")
            assert "Action item completed" in msg
            assert "Feature X" in msg
            assert "Completed" in msg

    def test_failed_completion_verification(self):
        # API was called but Slack List still returns completed=False
        items_store = [_test_robustness_make_item("item_1", "Feature X", completed=False)]

        def mock_complete_noop(item_id, ctx, list_id):
            return True

        with patch.object(slack_tools, "get_list_schema", return_value=_test_robustness_SCHEMA),\
             patch.object(slack_tools, "list_action_items", side_effect=lambda ctx, list_id: items_store),\
             patch.object(slack_tools, "complete_action_item", side_effect=mock_complete_noop),\
             patch.object(config, "has_permission", return_value=True):

            ctx = config.build_context(user_id="U12345", channel_id="C_DEV")
            parsed = {"intent": "complete", "task_name": "Feature X"}
            msg = main.handle_mutation(parsed, ctx, "test_mem")
            assert "still shows pending" in msg


class TestStaleThreadContext:
    def test_stale_item_filtered_out(self):
        item_1 = _test_robustness_make_item("item_1", "Old Task")
        item_2 = _test_robustness_make_item("item_2", "Current Task")
        ctx = config.build_context(user_id="U12345", channel_id="C_DEV")
        main.store_view("stale_mem", [item_1, item_2], schema=_test_robustness_SCHEMA, ctx=ctx)

        current_items = [item_2]
        parsed = {"intent": "complete", "selection": "first", "selection_index": 1}
        ctx = config.build_context(user_id="U12345", channel_id="C_DEV")

        with pytest.raises(ValueError, match="no longer exists"):
            main.resolve_targets(parsed, current_items, _test_robustness_SCHEMA, "stale_mem", ctx=ctx, intent="complete")


class TestUnassignedCreation:
    def test_create_without_assignee(self):
        created_items = []
        def mock_create(name, priority, assignee, due_date, ctx, list_id):
            it = _test_robustness_make_item("new_1", name, completed=False, user_id=assignee, priority=priority or "P3", due=due_date or "2026-09-25")
            created_items.append(it)
            return it

        with patch.object(slack_tools, "get_list_schema", return_value=_test_robustness_SCHEMA),\
             patch.object(slack_tools, "list_action_items", side_effect=lambda *args: list(created_items)),\
             patch.object(slack_tools, "create_action_item", side_effect=mock_create),\
             patch.object(config, "has_permission", return_value=True):

            ctx = config.build_context(user_id="U12345", channel_id="C_DEV")
            parsed = {"intent": "create", "task_name": "Documentation Update"}
            msg = main.handle_create(parsed, ctx)
            assert "Task created successfully" in msg
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
            _test_robustness_make_item("item_1", "Task Alpha", completed=False),
            _test_robustness_make_item("item_2", "Task Beta", completed=False),
        ]

        def mock_complete(item_id, ctx, list_id):
            for it in items_store:
                if it["id"] == item_id:
                    it["fields"][4]["checkbox"] = True
                    it["fields"][4]["value"] = True
            return True

        with patch.object(slack_tools, "get_list_schema", return_value=_test_robustness_SCHEMA),\
             patch.object(slack_tools, "list_action_items", side_effect=lambda ctx, list_id: [dict(x) for x in items_store]),\
             patch.object(slack_tools, "complete_action_item", side_effect=mock_complete),\
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
            assert all(slack_tools.extract_completed(x, _test_robustness_SCHEMA) for x in items_store)

    def test_positional_without_context_raises_clarification(self):
        main._pending.clear()
        items_store = [_test_robustness_make_item("item_1", "Lone Task", completed=False)]

        with patch.object(slack_tools, "get_list_schema", return_value=_test_robustness_SCHEMA),\
             patch.object(slack_tools, "list_action_items", return_value=items_store),\
             patch.object(config, "has_permission", return_value=True):

            # No "show tasks" executed in this thread
            res = main.process("I completed the last", "U12345", "C_DEV", "thread_unseen_999")
            assert "recently displayed action items in this thread" in res or "couldn't find" in res



# Migrated test coverage from test_thread_context.py
import pytest
from unittest.mock import patch
from src import app as main
from src import slack_client as slack_tools
from src import config

_test_thread_context_SCHEMA = {
    "schema": [
        {"id": "col_name", "key": "name", "name": "Task Name", "type": "text"},
        {"id": "col_assignee", "key": "assignee", "name": "Assignee", "type": "user"},
        {"id": "col_due", "key": "due_date", "name": "Due Date", "type": "date"},
        {"id": "col_priority", "key": "priority", "name": "Priority", "type": "select"},
        {"id": "col_completed", "key": "completed", "name": "Completed", "type": "checkbox"},
    ]
}

def _test_thread_context_make_item(item_id, name, completed=False, user_id="U12345", due="2026-09-20", priority="P2"):
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
        _test_thread_context_make_item("item_A", "Prepare project report", completed=False),
        _test_thread_context_make_item("item_B", "Complete Python assignment", completed=False),
        _test_thread_context_make_item("item_C", "Deploy production release", completed=False),
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

    with patch.object(slack_tools, "get_list_schema", return_value=_test_thread_context_SCHEMA),\
         patch.object(slack_tools, "list_action_items", side_effect=mock_list_action_items),\
         patch.object(slack_tools, "complete_action_item", side_effect=mock_complete_action_item),\
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
        assert slack_tools.extract_completed(items_store[0], _test_thread_context_SCHEMA) is False
        assert slack_tools.extract_completed(items_store[1], _test_thread_context_SCHEMA) is False
        assert slack_tools.extract_completed(items_store[2], _test_thread_context_SCHEMA) is True


def test_positional_references_all_variants():
    """
    Test first, second, third, last, both, all in threaded flows.
    """
    items_store = [
        _test_thread_context_make_item("item_1", "Task One", completed=False),
        _test_thread_context_make_item("item_2", "Task Two", completed=False),
        _test_thread_context_make_item("item_3", "Task Three", completed=False),
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

    with patch.object(slack_tools, "get_list_schema", return_value=_test_thread_context_SCHEMA),\
         patch.object(slack_tools, "list_action_items", side_effect=mock_list_action_items),\
         patch.object(slack_tools, "complete_action_item", side_effect=mock_complete_action_item),\
         patch.object(config, "has_permission", return_value=True):

        user_id = "U123"
        channel_id = "C_TEST"
        thread_ts = "1726000000.000999"

        # Show 3 tasks
        main.process("show tasks", user_id, channel_id, thread_ts=thread_ts)

        # Test 'first'
        res_first = main.process("complete first", user_id, channel_id, thread_ts=thread_ts)
        assert "Task One" in res_first
        assert slack_tools.extract_completed(items_store[0], _test_thread_context_SCHEMA) is True

        # Test 'second'
        res_second = main.process("complete the second", user_id, channel_id, thread_ts=thread_ts)
        assert "Task Two" in res_second
        assert slack_tools.extract_completed(items_store[1], _test_thread_context_SCHEMA) is True

        # Test 'third'
        res_third = main.process("complete the third", user_id, channel_id, thread_ts=thread_ts)
        assert "Task Three" in res_third
        assert slack_tools.extract_completed(items_store[2], _test_thread_context_SCHEMA) is True

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
        assert all(slack_tools.extract_completed(it, _test_thread_context_SCHEMA) for it in items_store)


def test_both_variant_with_two_items():
    """
    Test 'complete both' when exactly two tasks are displayed.
    """
    items_store = [
        _test_thread_context_make_item("item_1", "Task Alpha", completed=False),
        _test_thread_context_make_item("item_2", "Task Beta", completed=False),
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

    with patch.object(slack_tools, "get_list_schema", return_value=_test_thread_context_SCHEMA),\
         patch.object(slack_tools, "list_action_items", side_effect=mock_list_action_items),\
         patch.object(slack_tools, "complete_action_item", side_effect=mock_complete_action_item),\
         patch.object(config, "has_permission", return_value=True):

        user_id = "U123"
        channel_id = "C_TEST"
        thread_ts = "1726000000.000888"

        main.process("show tasks", user_id, channel_id, thread_ts=thread_ts)
        res_both = main.process("complete both", user_id, channel_id, thread_ts=thread_ts)
        assert "Task Alpha" in res_both
        assert "Task Beta" in res_both
        assert all(slack_tools.extract_completed(it, _test_thread_context_SCHEMA) for it in items_store)


def test_both_clarification_on_three_items():
    """Test 'both' prompts for clarification when more than 2 items exist."""
    items_store = [
        _test_thread_context_make_item("item_1", "Alpha Task", completed=False),
        _test_thread_context_make_item("item_2", "Beta Task", completed=False),
        _test_thread_context_make_item("item_3", "Gamma Task", completed=False),
    ]

    with patch.object(slack_tools, "get_list_schema", return_value=_test_thread_context_SCHEMA),\
         patch.object(slack_tools, "list_action_items", return_value=items_store),\
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
        _test_thread_context_make_item("item_A1", "Thread A Task One", completed=False),
        _test_thread_context_make_item("item_A2", "Thread A Task Two", completed=False),
    ]
    items_store_b = [
        _test_thread_context_make_item("item_B1", "Thread B Task One", completed=False),
        _test_thread_context_make_item("item_B2", "Thread B Task Two", completed=False),
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

    with patch.object(slack_tools, "get_list_schema", return_value=_test_thread_context_SCHEMA),\
         patch.object(slack_tools, "list_action_items", side_effect=mock_list_action_items),\
         patch.object(slack_tools, "complete_action_item", side_effect=mock_complete_action_item),\
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




# Migrated test coverage from test_local.py
from src import local


def test_local_entry_uses_shared_app_and_cleans_up(monkeypatch):
    events = []
    settings = type("Settings", (), {
        "app_token": "xapp-test",
        "validate": lambda self, socket_mode=False: events.append(("validate", socket_mode)),
    })()
    monkeypatch.setattr(local, "load_local_environment", lambda: events.append("dotenv"))
    monkeypatch.setattr(local.RuntimeConfig, "from_environment", lambda: settings)
    monkeypatch.setattr(local, "get_app", lambda: events.append("app") or object())
    monkeypatch.setattr(local, "start_local_services", lambda: events.append("start") or "scheduler")
    monkeypatch.setattr(local, "shutdown_local_services", lambda value: events.append(("stop", value)))

    class Handler:
        def __init__(self, app, token):
            assert token == "xapp-test"

        def start(self):
            events.append("socket")
            raise KeyboardInterrupt

    monkeypatch.setattr(local, "SocketModeHandler", Handler)
    local.main()
    assert events == ["dotenv", ("validate", True), "app", "start", "socket", ("stop", "scheduler")]



# Migrated test coverage from test_structure.py
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_project_structure_exists():
    required = (
        ".env.example", ".gitignore", "deploy.sh", "requirements.txt",
        "requirements-dev.txt", "README.md", "src/app.py", "src/handler.py",
        "src/local.py", "src/graph.py", "src/tools.py", "src/slack_client.py",
        "src/prompts.py", "src/config.py", "tests/conftest.py",
        "tests/test_tools.py", "tests/test_graph.py",
    )
    assert all((ROOT / path).is_file() for path in required)


def test_only_required_python_modules_remain_at_repository_depth_two():
    expected = {
        "src/app.py", "src/config.py", "src/graph.py", "src/handler.py",
        "src/local.py", "src/prompts.py", "src/slack_client.py", "src/tools.py",
        "tests/conftest.py", "tests/test_graph.py", "tests/test_tools.py",
    }
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.glob("*.py")
    } | {
        path.relative_to(ROOT).as_posix()
        for directory in ("src", "tests") for path in (ROOT / directory).glob("*.py")
    }
    assert actual == expected


@pytest.mark.parametrize("phrase,intent,detail", [
    ("give me my tasks", "list", ("assignee_self", True)),
    ("show task analytics", "visual_analytics", ("visualization_type", "priority")),
    ("show task analytics by priority", "visual_analytics", ("visualization_type", "priority")),
    ("visualize tasks by priority", "visual_analytics", ("visualization_type", "priority")),
    ("what is on the calendar today?", "calendar", ("calendar_mode", "today")),
    ("what deadlines are coming this week?", "calendar", ("calendar_mode", "week")),
    ("when can we meet?", "operations_intelligence", ("operations_mode", "meeting")),
    ("find a meeting time for the team", "operations_intelligence", ("operations_mode", "meeting")),
    ("I need you to figure out which tasks are putting our deployment at risk",
     "operations_intelligence", ("operations_mode", "risk")),
])
def test_regression_requests_route_deterministically(monkeypatch, phrase, intent, detail):
    from src import graph
    monkeypatch.setattr(graph, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("LLM called"))
    parsed = graph.parse_intent(phrase)
    assert parsed["intent"] == intent
    assert parsed[detail[0]] == detail[1]


def test_create_metadata_does_not_leak_into_title(monkeypatch):
    from src import graph
    monkeypatch.setattr(graph, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("LLM called"))
    parsed = graph.parse_intent(
        "Create a task called API review for Praveen with priority P2 due October 20")
    assert parsed["intent"] == "create"
    assert parsed["task_name"] == "API review"
    assert parsed["assignee"] == "Praveen"
    assert parsed["priority"] == "P2"
    assert parsed["due_date"] == "2026-10-20"


def test_incomplete_create_clause_cannot_become_task_title(monkeypatch):
    from src import graph
    monkeypatch.setattr(graph, "OLLAMA_API_KEY", "")
    assert graph.parse_intent("Prepare project report, assign")["intent"] != "create"


def test_secrets_and_generated_build_are_ignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignored
    assert "build/" in ignored
    assert "*.sqlite3-wal" in ignored


@pytest.mark.parametrize("phrase,intent", [
    ("list my task", "list"), ("list my tasks", "list"),
    ("show my tasks", "list"), ("list all task", "list"),
    ("list all tasks", "list"), ("show all tasks", "list"),
    ("list pending tasks", "list"), ("list completed tasks", "list"),
    ("focus today", "focus"), ("focus this week", "weekly_focus"),
    ("visualize my tasks", "visual_analytics"),
    ("visualize my all tasks", "visual_analytics"),
    ("visualize tasks by priority", "visual_analytics"),
    ("visualize tasks by assignee", "visual_analytics"),
    ("show task analytics", "visual_analytics"),
    ("show task distribution", "visual_analytics"),
    ("what is the weather today?", "out_of_scope"),
])
def test_required_deterministic_routes_never_call_ollama(monkeypatch, phrase, intent):
    from src import graph
    monkeypatch.setattr(graph, "_configured_ollama_client",
                        lambda *args, **kwargs: pytest.fail("LLM called"))
    assert graph.parse_intent(phrase)["intent"] == intent


@pytest.mark.parametrize("phrase,kind", [
    ("visualize tasks by assignee", "workload"),
    ("visualize tasks by status", "completion"),
    ("visualize tasks by priority", "priority"),
])
def test_visualization_dimension_is_preserved(phrase, kind):
    from src.graph import parse_intent
    assert parse_intent(phrase)["visualization_type"] == kind


def test_llm_create_title_with_command_grammar_is_rejected():
    from src.graph import validate_command
    with pytest.raises(ValueError, match="task name separately"):
        validate_command({"intent": "create", "task_name":
                          "Create a task called API review for Praveen with priority P2 due October 20"})
