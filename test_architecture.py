"""End-to-end offline tests through the real parser, resolver and Slack adapter."""
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
import logging
import re
from types import SimpleNamespace
from unittest.mock import Mock
import sqlite3

import pytest

import commands
import config
import delivery
import intent_parser
import main
import mutations
import progress_engine
import project_intelligence
import predictive_intelligence
import slack_tools
import content_ingestion
import action_item_extraction
import agent_orchestrator
import source_trace
import audit_log
import deadline_reminders
import slack_presentation
from references import parse_reference, select_ids, extract_contextual_reference
from references import TargetType


@pytest.mark.parametrize("phrase,mode", [
    ("give me an intelligence summary", "summary"),
    ("what risks are coming up?", "emerging_risks"),
    ("show deadline pressure", "deadline_pressure"),
    ("show workload outlook", "workload_outlook"),
])
def test_predictive_intelligence_routes_deterministically_without_llm(phrase, mode, monkeypatch):
    monkeypatch.setattr(intent_parser, "structured_model_json", lambda *a, **k: pytest.fail("LLM called"))
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == "intelligence_summary"
    assert parsed["intelligence_mode"] == mode


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


@pytest.mark.parametrize("phrase", [
    "command center", "show me the command center", "give me an overview",
    "give me an overview of our tasks", "give me a task overview",
    "show me what's happening with our tasks", "what is happening with our tasks?",
    "how are we doing?", "how are we doing with our action items?",
    "how are things going with our action items?", "show me the current task situation",
    "give me the current status", "summarize our action items",
])
def test_command_center_natural_overview_routes_without_llm(slack, phrase):
    slack.add("Critical work", assignee="UA", priority="P1",
              due=(main.current_date() - timedelta(days=1)).isoformat())
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == "command_center"
    response = ask(phrase)
    assert response.startswith("*Smart Task Command Center*")
    assert "📊 *Team Status*" in response
    assert "🔴 *Critical*" in response
    assert "*Autopilot*" in response
    assert "*No action has been taken automatically.*" in response
    assert "**" not in response


def test_command_center_counts_workload_and_prepares_safe_approval(slack, monkeypatch):
    today = main.current_date()
    slack.add("Late report", assignee="UA", priority="P1",
              due=(today - timedelta(days=1)).isoformat())
    slack.add("Upcoming API", assignee="UM", priority="P2",
              due=(today + timedelta(days=3)).isoformat())
    slack.add("Finished", completed=True, assignee="UA", priority="P3")
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: sent.append(args))
    response = ask("command center")
    assert "• 2 pending · 1 completed" in response
    assert "• 1 P1 · 1 P2 · 0 P3 among pending tasks" in response
    assert "*Prepared Message*" in response
    assert "`send sentinel alert 1`" in response
    assert not sent
    assert "Sentinel Action Completed" in ask("send sentinel alert 1")
    assert len(sent) == 1


def test_intelligence_summary_uses_one_authorized_snapshot_and_never_mutates(slack, monkeypatch):
    today = date(2026, 9, 29)
    monkeypatch.setattr(main, "current_date", lambda: today)
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: {"UA": "AasthaA", "UM": "Praveen"}.get(user_id))
    slack.add("Late P1", assignee="UA", priority="P1",
              due=(today - timedelta(days=1)).isoformat())
    slack.add("Soon P1", assignee="UM", priority="P1",
              due=(today + timedelta(days=1)).isoformat())
    slack.add("Unassigned", priority="P2", due=(today + timedelta(days=5)).isoformat())
    response = ask("give me an intelligence summary", thread="PREDICTIVE_SUMMARY")
    assert response.startswith("*🧠 Task Intelligence*")
    assert "3 pending · 0 completed · 1 overdue" in response
    assert "2 P1 · 1 P2 · 0 P3" in response
    assert "Soon P1" in response
    assert "_No actions were taken._" in response
    # Existing schema discovery and item retrieval each use one Lists read;
    # all intelligence sections then reuse the resulting normalized snapshot.
    assert len(slack.list_requests) == 2
    assert not slack.writes
    assert "UA" not in response and "UM" not in response


def test_predictive_risk_empty_state_is_truthful_and_read_only(slack, monkeypatch):
    today = date(2026, 9, 29)
    monkeypatch.setattr(main, "current_date", lambda: today)
    slack.add("Routine future work", assignee="UA", priority="P3",
              due=(today + timedelta(days=10)).isoformat())
    response = ask("what risks are coming up?", thread="PREDICTIVE_EMPTY")
    assert "No emerging risks were identified" in response
    assert "_No changes were made._" in response
    assert not slack.writes


def test_command_center_member_scope_does_not_leak_other_tasks(slack):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    slack.add("Morgan private", assignee="UM", priority="P1", due=yesterday)
    slack.add("Alex private", assignee="UA", priority="P1", due=yesterday)
    response = ask("command center", user="UM")
    assert "Morgan private" in response
    assert "Alex private" not in response


def test_command_center_owner_risk_explanation_uses_real_member_tasks(slack):
    today = main.current_date()
    slack.add("Praveen release", assignee="UP", priority="P1", due=today.isoformat())
    response = ask("why is Praveen at risk?")
    assert "Risk Explanation" in response
    assert "Praveen release" in response
    assert "P1" in response
    assert "No action has been taken automatically" in response
    assert "U" + "P" not in response


def test_command_center_contextual_followup_and_message_preparation_are_safe(slack):
    slack.add("Praveen late release", assignee="UP", priority="P1",
              due=(main.current_date() - timedelta(days=1)).isoformat())
    explanation = ask("why is Praveen at risk?", thread="CC_FOLLOWUP")
    assert "Praveen late release" in explanation
    followup = ask("what should we do about it?", thread="CC_FOLLOWUP")
    assert "Recommended Action" in followup
    prepared = ask("prepare the message", thread="CC_FOLLOWUP")
    assert "*Prepared Message*" in prepared
    assert "`send sentinel alert 1`" in prepared
    assert "No action has been taken automatically" in prepared


def test_command_center_contextual_followup_requires_prior_context(slack):
    response = ask("prepare the message", thread="CC_NO_CONTEXT")
    assert "don't have an active Command Center risk" in response


def test_ambiguous_send_it_never_executes_an_action(slack):
    response = ask("send it", thread="CC_SEND_SAFE")
    assert response == "Please approve a displayed recommendation with `send sentinel alert N`."


@pytest.mark.parametrize("phrase", [
    "why does that need attention?", "why does this need attention?",
    "why is this risky?", "what should I do about this task?",
])
def test_ranked_focus_pronoun_resolves_to_primary_task(slack, phrase):
    today = main.current_date()
    slack.add("Primary urgent task", assignee="UA", priority="P1",
              due=(today - timedelta(days=1)).isoformat())
    slack.add("Later task", assignee="UA", priority="P2",
              due=(today + timedelta(days=2)).isoformat())
    ask("what should I do next?", thread="CC_TASK_PRONOUN")
    response = ask(phrase, thread="CC_TASK_PRONOUN")
    assert "Primary urgent task" in response
    assert "Later task" not in response
    assert "couldn't resolve the Slack user" not in response


def test_contextual_task_can_prepare_but_not_send_message_automatically(slack):
    slack.add("Context reminder", assignee="UA", priority="P1",
              due=(main.current_date() - timedelta(days=1)).isoformat())
    ask("what should I do next?", thread="CC_TASK_MESSAGE")
    response = ask("prepare a message for it", thread="CC_TASK_MESSAGE")
    assert "*Prepared Message*" in response
    assert "`send sentinel alert 1`" in response
    assert "No action has been taken automatically" in response
    assert not slack.writes


def test_ambiguous_task_pronoun_lists_choices_without_guessing(slack):
    slack.add("First possible task", assignee="UA")
    slack.add("Second possible task", assignee="UA")
    ask("list all tasks", thread="CC_AMBIGUOUS_TASK")
    response = ask("why does that need attention?", thread="CC_AMBIGUOUS_TASK")
    assert response == ("I still need to know which task you mean: "
                        "First possible task, or Second possible task.")


def _start_pending_task_clarification(slack, thread, *, two_overdue=False):
    today = main.current_date()
    slack.add("Follow-up Test Task", assignee="UA", priority="P1",
              due=(today - timedelta(days=2)).isoformat())
    slack.add("Reminder Docs Task", assignee="UA", priority="P2",
              due=((today - timedelta(days=1)) if two_overdue
                   else (today + timedelta(days=2))).isoformat())
    slack.add("Prepare client presentation", assignee="UA", priority="P3",
              due=(today + timedelta(days=4)).isoformat())
    ask("list all tasks", thread=thread)
    return ask("why does that need attention?", thread=thread)


def test_pending_task_clarification_survives_related_followup(slack):
    first = _start_pending_task_clarification(slack, "CC_PENDING_FOLLOWUP")
    assert "Follow-up Test Task" in first and "Reminder Docs Task" in first
    response = ask("what should we do about it?", thread="CC_PENDING_FOLLOWUP")
    assert "I still need to know which task you mean" in response
    assert "active Command Center risk" not in response


@pytest.mark.parametrize("answer", ["Follow-up Test Task", "1", "the first one"])
def test_pending_task_clarification_resolves_name_number_or_ordinal(slack, answer):
    thread = "CC_RESOLVE_" + re.sub(r"\W+", "_", answer)
    _start_pending_task_clarification(slack, thread)
    response = ask(answer, thread=thread)
    assert "Follow-up Test Task" in response
    assert "Overdue by 2 days" in response
    assert "Recommended Action" in response
    # Resolution clears the pending question and establishes a safe task referent.
    followup = ask("why is this risky?", thread=thread)
    assert "Follow-up Test Task" in followup


def test_pending_task_clarification_resolves_unique_overdue_description(slack):
    _start_pending_task_clarification(slack, "CC_RESOLVE_OVERDUE")
    response = ask("the overdue one", thread="CC_RESOLVE_OVERDUE")
    assert "Follow-up Test Task" in response and "Overdue by 2 days" in response


def test_pending_task_clarification_rejects_ambiguous_overdue_description(slack):
    _start_pending_task_clarification(slack, "CC_AMBIGUOUS_OVERDUE", two_overdue=True)
    response = ask("the overdue one", thread="CC_AMBIGUOUS_OVERDUE")
    assert "which overdue task" in response
    assert "Follow-up Test Task" in response and "Reminder Docs Task" in response


def test_pending_clarification_cannot_be_resolved_by_another_user(slack):
    _start_pending_task_clarification(slack, "CC_CLARIFY_PRIVATE")
    response = ask("1", user="UM", thread="CC_CLARIFY_PRIVATE")
    assert "Follow-up Test Task" not in response


@pytest.mark.parametrize("phrase,kind,mode", [
    ("visualize our workload", "workload", "chart"),
    ("visualise our workload", "workload", "chart"),
    ("give me a chart of our workload", "workload", "chart"),
    ("graph our workload", "workload", "chart"),
    ("plot our workload", "workload", "chart"),
    ("show workload visually", "workload", "chart"),
    ("give me a visual summary of our workload", "workload", "chart"),
    ("give me a chart of our tasks", "dashboard", "dashboard"),
    ("show task distribution", "priority", "chart"),
    ("show priorities visually", "priority", "chart"),
    ("visualize completion", "completion", "chart"),
    ("visualize deadlines", "deadlines", "chart"),
    ("show me a dashboard", "dashboard", "dashboard"),
])
def test_visual_intents_are_deterministic_without_llm(phrase, kind, mode, monkeypatch):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("visual routing must not call the LLM")))
    parsed = commands.validate_command(intent_parser.parse_intent(phrase))
    assert parsed["intent"] == "visual_analytics"
    assert parsed["visualization_type"] == kind and parsed["response_mode"] == mode


def test_visual_workload_uses_authorized_snapshot_values(slack):
    slack.add("A one", assignee="UA")
    slack.add("A two", assignee="UA")
    slack.add("P one", assignee="UP")
    response = ask("visualize our workload", thread="VISUAL_WORKLOAD")
    assert isinstance(response, dict)
    assert response["visual"]["filename"] == "task-workload-bar.png"
    image = main.base64.b64decode(response["visual"]["content_base64"])
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert "Alex" in response["text"]


def test_visual_workload_respects_member_rbac(slack):
    slack.add("Member visible", assignee="UM")
    slack.add("Admin private", assignee="UA")
    response = ask("visualize our workload", user="UM", thread="VISUAL_RBAC")
    assert isinstance(response, dict)
    assert "Morgan" in response["text"]
    assert "Alex" not in response["text"] and "Admin private" not in response["text"]


def test_visual_empty_state_does_not_generate_chart(slack):
    response = ask("visualize our workload", thread="VISUAL_EMPTY")
    assert isinstance(response, str)
    assert "No meaningful authorized task data" in response


def test_visual_generation_failure_falls_back_to_factual_text(slack, monkeypatch):
    slack.add("Fallback", assignee="UA")
    monkeypatch.setattr(main.visual_analytics, "render_chart_png",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("chart failure")))
    response = ask("visualize our workload", thread="VISUAL_FAILURE")
    assert isinstance(response, str)
    assert "Pending Workload by Owner" in response and "Alex: 1" in response


def test_visual_followups_change_dimension_then_return_to_text_risk(slack):
    today = main.current_date()
    slack.add("Urgent visual", assignee="UA", priority="P1",
              due=(today - timedelta(days=1)).isoformat())
    slack.add("Later visual", assignee="UM", priority="P2",
              due=(today + timedelta(days=3)).isoformat())
    workload_text = ask("show me our workload", thread="VISUAL_FOLLOWUP")
    workload = ask("visualize it", thread="VISUAL_FOLLOWUP")
    priorities = ask("what about priorities?", thread="VISUAL_FOLLOWUP")
    deadlines = ask("and deadlines?", thread="VISUAL_FOLLOWUP")
    attention = ask("which one needs attention?", thread="VISUAL_FOLLOWUP")
    assert isinstance(workload_text, str) and "Team Workload" in workload_text
    assert workload["visual"]["filename"].endswith("workload-bar.png")
    assert priorities["visual"]["filename"].endswith("priority-pie.png")
    assert deadlines["visual"]["filename"].endswith("deadlines-bar.png")
    assert isinstance(attention, str) and "Action Item Sentinel" in attention


@pytest.mark.parametrize("phrase,expected_intent", [
    ("how many tasks do I have?", "list"),
    ("why is Praveen at risk?", "command_center"),
    ("what should I do next?", "list"),
    ("show me the overdue task", "list"),
])
def test_text_first_requests_do_not_route_to_visuals(phrase, expected_intent):
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == expected_intent
    assert parsed.get("response_mode") is None


def test_plain_workload_request_uses_existing_text_mode_without_llm(slack, monkeypatch):
    slack.add("Text workload", assignee="UA")
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("plain workload should be deterministic")))
    parsed = commands.validate_command(intent_parser.parse_intent("show me our workload"))
    assert parsed["intent"] == "workload" and parsed["response_mode"] == "text"
    response = ask("show me our workload", thread="WORKLOAD_TEXT_MODE")
    assert isinstance(response, str) and "Team Workload" in response


def test_workload_response_mode_logging_is_explicit(slack, caplog):
    caplog.set_level("INFO")
    slack.add("Logged workload", assignee="UA")
    ask("show me our workload", thread="WORKLOAD_LOG_TEXT")
    ask("visualize our workload", thread="WORKLOAD_LOG_CHART")
    assert "visual_request intent=workload response_mode=text visualization_type=none" in caplog.text
    assert "visual_request intent=workload response_mode=chart visualization_type=bar" in caplog.text


@pytest.mark.parametrize("phrase,kind,chart_type", [
    ("show workload as a bar chart", "workload", "bar"),
    ("show our workload as a pie chart", "workload", "pie"),
    ("visualize task priorities", "priority", "auto"),
    ("show priority distribution as a pie", "priority", "pie"),
    ("visualize upcoming deadlines", "deadlines", "auto"),
    ("visualize completion", "completion", "auto"),
    ("show task completion over time", "completed_trend", "line"),
    ("give me a table of overdue tasks", "overdue_tasks", "table"),
])
def test_explicit_chart_type_routing(phrase, kind, chart_type):
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == "visual_analytics"
    assert parsed["visualization_type"] == kind and parsed["chart_type"] == chart_type


def test_visual_chart_type_followups_reuse_current_distribution(slack):
    slack.add("Owner one", assignee="UA", priority="P1")
    slack.add("Owner two", assignee="UM", priority="P2")
    first = ask("visualize our workload", thread="VISUAL_TYPE_FOLLOWUP")
    pie = ask("show that as a pie", thread="VISUAL_TYPE_FOLLOWUP")
    table = ask("put that in a table", thread="VISUAL_TYPE_FOLLOWUP")
    assert first["visual"]["filename"].endswith("workload-bar.png")
    assert pie["visual"]["filename"].endswith("workload-pie.png")
    assert table["visual"]["filename"].endswith("workload-table.png")


def test_historical_line_request_without_history_falls_back_truthfully(slack):
    slack.add("No completion timestamp", completed=True, assignee="UA")
    response = ask("show task completion over time", thread="VISUAL_NO_HISTORY")
    assert isinstance(response, str)
    assert "No meaningful authorized task data" in response


def test_overdue_table_uses_real_authorized_tasks(slack):
    slack.add("Late table task", assignee="UA", priority="P1",
              due=(main.current_date() - timedelta(days=1)).isoformat())
    slack.add("Future table task", assignee="UA", priority="P2",
              due=(main.current_date() + timedelta(days=1)).isoformat())
    response = ask("give me a table of overdue tasks", thread="VISUAL_TABLE")
    assert response["visual"]["filename"] == "task-overdue_tasks.png"
    assert "1 matching task" in response["text"]


def test_visual_response_uploads_in_memory_png_without_exposing_paths(monkeypatch):
    calls = []

    class Client:
        def chat_postMessage(self, **kwargs):
            calls.append(("message", kwargs))
            return {"ok": True, "ts": "123.4"}

        def files_upload_v2(self, **kwargs):
            path = Path(kwargs["file"])
            assert path.is_file()
            kwargs["uploaded_bytes"] = path.read_bytes()
            calls.append(("upload", kwargs))
            return {"ok": True}

    monkeypatch.setattr(main, "app", SimpleNamespace(client=Client()))
    response = {"text": "Factual fallback", "visual": {
        "content_base64": main.base64.b64encode(b"\x89PNG\r\n\x1a\nexact").decode("ascii"),
        "filename": "task-workload-bar.png",
        "title": "Pending Workload"}}
    main.send_and_record("C", response, thread_ts="T")
    upload = next(value for kind, value in calls if kind == "upload")
    assert upload["uploaded_bytes"].startswith(b"\x89PNG") and upload["thread_ts"] == "T"
    assert upload["initial_comment"] == "Factual fallback"
    assert not Path(upload["file"]).exists()


def test_visual_upload_failure_keeps_delivered_text_fallback(monkeypatch):
    posted = []

    class Client:
        def chat_postMessage(self, **kwargs):
            posted.append(kwargs)
            return {"ok": True}

        def files_upload_v2(self, **kwargs):
            raise RuntimeError("upload unavailable")

    monkeypatch.setattr(main, "app", SimpleNamespace(client=Client()))
    response = {"text": "Short chart caption", "fallback_text": "Useful factual text", "visual": {
        "content_base64": main.base64.b64encode(b"png").decode("ascii"),
        "filename": "chart.png", "title": "Chart"}}
    assert main.send_and_record("C", response) == {"ok": True}
    assert "couldn't upload" in posted[0]["text"]
    assert "Useful factual text" in posted[0]["text"]


def test_visual_upload_permission_error_logs_safe_slack_diagnostics(monkeypatch, caplog):
    posted = []

    class UploadError(Exception):
        response = {
            "error": "missing_scope", "needed": "files:write",
            "provided": "files:read,chat:write",
        }

    class Client:
        @staticmethod
        def chat_postMessage(**kwargs):
            posted.append(kwargs)
            return {"ok": True}

        @staticmethod
        def files_upload_v2(**kwargs):
            raise UploadError("Slack rejected upload")

    caplog.set_level("INFO")
    monkeypatch.setattr(main, "app", SimpleNamespace(client=Client()))
    response = {"text": "Chart caption", "fallback_text": "Workload summary", "visual": {
        "content_base64": main.base64.b64encode(b"\x89PNG\r\n\x1a\nvalid").decode("ascii"),
        "filename": "chart.png", "title": "Chart"}}
    main.send_and_record("C", response, thread_ts="T")

    assert "error_code=missing_scope" in caplog.text
    assert "needed_scope=files:write" in caplog.text
    assert "fallback=text" in caplog.text
    assert posted and "couldn't upload" in posted[0]["text"]


@pytest.mark.parametrize("phrase,expected_type", [
    ("visualize our workload", "bar"),
    ("show our workload as a pie chart", "pie"),
])
def test_visual_workload_pipeline_generates_file_and_uploads_to_slack(
        slack, monkeypatch, phrase, expected_type):
    slack.add("Aastha work", assignee="UA")
    slack.add("Praveen work", assignee="UP")
    generated = []
    uploaded = []
    original = main.visual_analytics.render_chart_png

    def render(dataset, chart_type):
        generated.append((dataset.series, chart_type))
        return original(dataset, chart_type)

    def upload(**kwargs):
        path = Path(kwargs["file"])
        uploaded.append({**kwargs, "content": path.read_bytes(), "exists": path.exists()})
        return {"ok": True}

    class Client:
        files_upload_v2 = staticmethod(upload)

        @staticmethod
        def chat_postMessage(**kwargs):
            raise AssertionError("successful visualization delivery must upload the chart")

    monkeypatch.setattr(main.visual_analytics, "render_chart_png", render)
    monkeypatch.setattr(main, "app", SimpleNamespace(client=Client()))
    response = ask(phrase, thread=f"VISUAL_DELIVERY_{expected_type}")
    main.send_and_record("C", response, thread_ts="THREAD")

    assert generated and generated[0][1] == expected_type
    assert uploaded and uploaded[0]["exists"]
    assert uploaded[0]["content"].startswith(b"\x89PNG\r\n\x1a\n")
    assert uploaded[0]["filename"].endswith(f"-{expected_type}.png")
    assert uploaded[0]["thread_ts"] == "THREAD"
    assert not Path(uploaded[0]["file"]).exists()


def test_plain_workload_delivery_does_not_attempt_file_upload(slack, monkeypatch):
    slack.add("Text-only workload", assignee="UA")
    posted = []

    class Client:
        @staticmethod
        def chat_postMessage(**kwargs):
            posted.append(kwargs)
            return {"ok": True, "ts": "1.2"}

        @staticmethod
        def files_upload_v2(**kwargs):
            raise AssertionError("plain workload response must not upload a file")

    response = ask("show me our workload", thread="WORKLOAD_NO_UPLOAD")
    monkeypatch.setattr(main, "app", SimpleNamespace(client=Client()))
    main.send_and_record("C", response, thread_ts="THREAD")
    assert posted and "Team Workload" in posted[0]["text"]


def test_task_pronoun_context_is_isolated_by_user_and_thread(slack):
    slack.add("Private contextual task", assignee="UA", priority="P1",
              due=(main.current_date() - timedelta(days=1)).isoformat())
    ask("what should I do next?", user="UA", thread="CC_PRIVATE")
    other_user = ask("why does that need attention?", user="UM", thread="CC_PRIVATE")
    other_thread = ask("why does that need attention?", user="UA", thread="CC_OTHER")
    assert "Private contextual task" not in other_user
    assert "Private contextual task" not in other_thread
    assert "clear recent task" in other_user and "clear recent task" in other_thread


@pytest.mark.parametrize("phrase,period", [
    ("what changed?", "today"),
    ("anything different?", "today"),
    ("anything different with our tasks?", "today"),
    ("did anything change?", "today"),
    ("has anything changed?", "today"),
    ("what is different now?", "today"),
    ("any updates?", "today"),
    ("any task updates?", "today"),
    ("what happened since the last check?", "since_last_check"),
])
def test_change_digest_variations_are_deterministic(phrase, period):
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == "history"
    assert parsed["history_period"] == period


@pytest.mark.parametrize("phrase", [
    "what should I do next?", "what is the most important thing right now?",
    "what should we handle first?", "what are my next steps?", "prepare next steps",
])
def test_next_step_variations_reuse_focus_intelligence(phrase):
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == "list"
    assert parsed["focus_intelligence"] is True
    assert parsed["assignee_self"] is True


def test_sentinel_personal_scope_is_rbac_filtered_and_slack_native(slack):
    today = main.current_date()
    slack.add("Morgan urgent", assignee="UM", priority="P1", due=today.isoformat())
    slack.add("Alex urgent", assignee="UA", priority="P1", due=today.isoformat())
    response = ask("what requires my attention?", user="UM")
    assert "Morgan urgent" in response
    assert "Alex urgent" not in response
    assert "**" not in response
    assert "UM" not in response


def test_sentinel_team_scope_and_human_approved_reminder(slack, monkeypatch):
    today = main.current_date()
    slack.add("Deadline review", assignee="UA", priority="P1", due=today.isoformat())
    response = ask("show sentinel alerts")
    assert "Deadline review" in response
    assert "send sentinel alert 1" in response
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder",
                        lambda recipient, message: sent.append((recipient, message)))
    approved = ask("send sentinel alert 1")
    assert "Reminder sent" in approved
    assert sent and sent[0][0] == "UA"
    assert "Deadline review" in sent[0][1]
    assert "**" not in sent[0][1]
    repeated = ask("send sentinel alert 1").lower()
    assert "already" in repeated or "no longer active" in repeated


def test_sentinel_viewer_cannot_approve_follow_up(slack):
    today = main.current_date()
    slack.add("Viewer task", assignee="UV", priority="P1", due=today.isoformat())
    assert "Viewer task" in ask("what requires my attention?", user="UV")
    response = ask("send sentinel alert 1", user="UV")
    assert "cannot approve" in response.lower()
    assert not slack.writes


def test_sentinel_separates_task_risks_from_workload_signals(slack):
    today = main.current_date()
    yesterday = (today - timedelta(days=1)).isoformat()
    tomorrow = (today + timedelta(days=1)).isoformat()
    slack.add("Overdue review", assignee="UA", priority="P1", due=yesterday)
    slack.add("Upcoming review", assignee="UA", priority="P1", due=tomorrow)
    response = ask("what requires attention?")
    assert "🔴 *Overdue*" in response
    assert "🟠 *Deadline Risk*" in response
    assert "👥 *Workload Risk*" in response
    assert "Overdue review" in response
    assert "Upcoming review" in response
    assert "Multiple urgent action items" not in response
    assert "**Action Item Sentinel**" not in response
    assert response.startswith("*Action Item Sentinel*")
    assert "*Smart Task Autopilot*" in response
    assert "*Prepared Message*" in response
    assert "Overdue review" in response
    assert "UA" not in response


def test_named_sentinel_risk_explanation_preserves_task_title(slack):
    today = main.current_date()
    slack.add("Follow-up Test Task", assignee="UA", priority="P1", due=today.isoformat())
    parsed = intent_parser.parse_intent("why is Follow-up Test Task risky?")
    assert parsed["intent"] == "sentinel"
    assert parsed["task_name"] == "Follow-up Test Task"
    response = ask("why is Follow-up Test Task risky?")
    assert "Follow-up Test Task" in response
    assert "P1 priority" in response


def test_workload_only_sentinel_response_includes_non_executable_autopilot(slack):
    due = (main.current_date() + timedelta(days=2)).isoformat()
    slack.add("First concentrated task", assignee="UA", priority="P1", due=due)
    slack.add("Second concentrated task", assignee="UA", priority="P1", due=due)
    response = ask("what requires attention?")
    assert "👥 *Workload Risk*" in response
    assert "*Smart Task Autopilot*" in response
    assert "review the 2 P1 tasks and confirm their deadlines" in response
    assert "Human review is required" in response
    assert "No action has been taken automatically" in response
    assert "Multiple urgent action items" not in response
    assert "send sentinel alert" not in response
    assert "**Smart Task Autopilot**" not in response


def test_sentinel_clean_format_does_not_execute_and_uses_blockquote(slack, monkeypatch):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    slack.add("Polished reminder", assignee="UA", priority="P1", due=yesterday)
    monkeypatch.setattr(
        main, "_send_deadline_reminder",
        lambda *args: (_ for _ in ()).throw(AssertionError("detection must not send")))
    response = ask("what requires attention?")
    assert "*Action Item Sentinel*" in response
    assert "🔴 *Overdue*" in response
    assert "*Smart Task Autopilot*" in response
    assert "*Recommended Action*" in response
    assert "*Prepared Message*\n> " in response
    assert "*Approval*\n`send sentinel alert 1`" in response
    assert "*No action has been taken automatically.*" in response
    assert "**" not in response


def test_sentinel_numbering_executes_exact_displayed_alert(slack, monkeypatch):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    slack.add("First alert", assignee="UA", priority="P1", due=yesterday)
    slack.add("Second alert", assignee="UM", priority="P1", due=yesterday)
    response = ask("what requires attention?")
    assert "1. *First alert*" in response
    assert "2. *Second alert*" in response
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder",
                        lambda recipient, message: sent.append((recipient, message)))
    assert "Sentinel Action Completed" in ask("send sentinel alert 2")
    assert len(sent) == 1
    assert sent[0][0] == "UM"
    assert "Second alert" in sent[0][1]
    assert "First alert" not in sent[0][1]


def test_multiple_workload_recommendations_are_grouped_concisely(slack):
    due = (main.current_date() + timedelta(days=2)).isoformat()
    for owner, prefix in (("UA", "Alex"), ("UM", "Morgan")):
        slack.add(f"{prefix} one", assignee=owner, priority="P1", due=due)
        slack.add(f"{prefix} two", assignee=owner, priority="P1", due=due)
    response = ask("detect emerging risks")
    assert response.count("*Smart Task Autopilot*") == 1
    assert response.count("*Recommended Action*") == 1
    assert response.count("review the 2 P1 tasks and confirm their deadlines") == 2
    assert "*Prepared Message*" not in response
    assert "send sentinel alert" not in response


def test_sentinel_empty_state_distinguishes_pending_without_risk(slack):
    due = (main.current_date() + timedelta(days=30)).isoformat()
    slack.add("Safe pending task", assignee="UA", priority="P3", due=due)
    response = ask("what requires attention?")
    assert response.startswith("*Action Item Sentinel*")
    assert "*No active risks found.*" in response
    assert "pending action items" in response
    assert "No action items found" not in response


def test_sentinel_rejects_completed_or_missing_task_after_preview(slack, monkeypatch):
    today = main.current_date()
    item = slack.add("State changed", assignee="UA", priority="P1", due=today.isoformat())
    ask("detect emerging risks")
    item["fields"][1]["checkbox"] = True
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: sent.append(args))
    response = ask("send alert 1")
    assert "Sentinel Action Not Completed" in response
    assert "task state has changed" in response
    assert not sent

    item["fields"][1]["checkbox"] = False
    ask("detect emerging risks")
    slack.items.remove(item)
    response = ask("send alert 1")
    assert "Sentinel Action Not Completed" in response
    assert "no longer exists" in response
    assert not sent


def test_sentinel_dismiss_is_persistent_and_idempotent(slack, monkeypatch):
    today = main.current_date()
    slack.add("Dismiss me", assignee="UA", priority="P1", due=today.isoformat())
    ask("detect emerging risks")
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: sent.append(args))
    first = ask("dismiss alert 1")
    second = ask("dismiss alert 1")
    assert "Sentinel Action Completed" in first
    assert "No reminder was sent" in first
    assert "already been handled" in second
    assert not sent


def test_sentinel_rejects_expired_and_cross_user_approval(slack, monkeypatch):
    today = main.current_date()
    slack.add("Approval scope", assignee="UA", priority="P1", due=today.isoformat())
    ask("detect emerging risks")
    with sqlite3.connect(main.DB_PATH) as connection:
        connection.execute("UPDATE sentinel_displays SET displayed_at=0")
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: sent.append(args))
    assert "approval has expired" in ask("send reminder for alert 1")
    assert "not shown to you" in ask("send reminder for alert 1", user="UV")
    assert not sent


def test_sentinel_invalid_alert_number_is_safe(slack, monkeypatch):
    today = main.current_date()
    slack.add("Numbered risk", assignee="UA", priority="P1", due=today.isoformat())
    ask("detect emerging risks")
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: sent.append(args))
    response = ask("send sentinel alert 99")
    assert "Sentinel Action Not Completed" in response
    assert "not shown" in response
    assert not sent


def test_follow_up_sender_posts_directly_to_user_and_requires_confirmation(monkeypatch):
    calls = []
    client = SimpleNamespace(
        chat_postMessage=lambda **kwargs: calls.append(kwargs) or
        {"ok": True, "channel": "D_OWNER", "ts": "123.456"},
        conversations_open=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("conversations.open must not be required")),
    )
    monkeypatch.setattr(main, "app", SimpleNamespace(client=client))
    result = main._send_deadline_reminder("UA", "**Reminder**")
    assert result["ts"] == "123.456"
    assert calls == [{"channel": "UA", "text": "*Reminder*"}]


@pytest.mark.parametrize("response", [
    {"ok": False, "error": "channel_not_found"},
    {"ok": True, "channel": "D_OWNER"},
])
def test_follow_up_sender_rejects_unconfirmed_delivery(monkeypatch, response):
    calls = []
    client = SimpleNamespace(
        chat_postMessage=lambda **kwargs: calls.append(kwargs) or response)
    monkeypatch.setattr(main, "app", SimpleNamespace(client=client))
    with pytest.raises(RuntimeError):
        main._send_deadline_reminder("UA", "Reminder")
    assert len(calls) == 1


def test_sentinel_sender_failure_keeps_alert_active_then_success_is_once(slack, monkeypatch):
    today = main.current_date()
    slack.add("Retry safely", assignee="UA", priority="P1", due=today.isoformat())
    ask("detect emerging risks")
    attempts = []

    def fail_once(recipient, message):
        attempts.append((recipient, message))
        raise RuntimeError("simulated Slack rejection")

    monkeypatch.setattr(main, "_send_deadline_reminder", fail_once)
    failed = ask("send alert 1")
    assert "alert remains active" in failed
    store = main._sentinel_engine().store
    alert_id, _ = store.resolve_display("L", "C", "UA", 1)
    assert store.alert(alert_id).status == "active"

    monkeypatch.setattr(main, "_send_deadline_reminder",
                        lambda recipient, message: attempts.append((recipient, message)))
    assert "Sentinel Action Completed" in ask("send alert 1")
    assert store.alert(alert_id).status == "approved"
    assert "already been handled" in ask("send alert 1")
    assert len(attempts) == 2


@pytest.mark.parametrize("changed_field,new_value", [
    ("priority", "priority_2"),
    ("due", "2099-12-31"),
    ("owner", "UM"),
])
def test_sentinel_rejects_material_task_changes(slack, monkeypatch, changed_field, new_value):
    today = main.current_date()
    item = slack.add("Material change", assignee="UA", priority="P1", due=today.isoformat())
    ask("detect emerging risks")
    field = next(value for value in item["fields"] if value["column_id"] == changed_field)
    if changed_field == "priority":
        field["select"] = [new_value]
    elif changed_field == "due":
        field["date"] = [new_value]
    else:
        field["user"] = [new_value]
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: sent.append(args))
    response = ask("send alert 1")
    assert "task state has changed" in response
    assert not sent


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


def test_professional_list_and_focus_responses_preserve_authorized_scope(slack):
    today = main.current_date().isoformat()
    slack.add("Mine later", assignee="UA", due=(main.current_date() + timedelta(days=2)).isoformat())
    slack.add("Someone else's today", assignee="UM", due=today)
    mine = ask("list my task")
    assert mine.startswith("*📋 My Action Items*\n\n*1 pending*")
    assert "Aastha" not in mine and "· Pending" not in mine
    assert "Mine later" in mine and "Someone else's today" not in mine
    focus = ask("what should I focus on today")
    assert focus.startswith("*🎯 Focus Today*")
    assert "No action items are due today" in focus
    assert "*Pending work:* 1 task" in focus


def test_list_all_tasks_uses_pending_and_completed_sections(slack):
    slack.add("Open work")
    slack.add("Closed work", completed=True)
    response = ask("list all tasks")
    assert response.startswith("*📋 Action Items*\n\n*2 tasks*")
    assert "*Pending*" in response and "*Completed*" in response
    assert "Open work" in response and "Closed work" in response


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
    assert re.search(r"Pending\s+2", response)
    assert re.search(r"Completed\s+1", response)


def test_empty_status_comparison_renders_zero_counts(slack):
    response = ask("compare finished versus unfinished items")
    assert re.search(r"Pending\s+0", response)
    assert re.search(r"Completed\s+0", response)


@pytest.mark.parametrize("phrase,expected_due", [
    ("create release checks due 2026-09-30", "2026-09-30"),
    ("add release checks to date 2026-09-30", "2026-09-30"),
    ("create release checks for September 30", "2026-09-30"),
    ("add release checks due next Friday", None),
])
def test_creation_date_clause_is_a_field_not_title_or_assignee(slack, phrase, expected_due):
    expected_due = expected_due or intent_parser._resolve_natural_date(
        "next Friday", main.current_date())
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


def test_create_duplicate_and_update_use_professional_templates_without_raw_ids(slack):
    created = ask("create Client Report priority P3")
    assert created.startswith("*✓ Action Item Created*")
    assert "*Client Report*" in created
    assert "P3 · Unassigned · No due date · Pending" in created
    assert "created successfully and verified in *Action Items*" in created
    duplicate = ask("create Client Report priority P3")
    assert duplicate.startswith("*Task already exists*")
    assert "No changes were made" in duplicate
    updated = ask("change Client Report priority to P1")
    assert updated.startswith("*✓ Action item updated*")
    assert "Priority: P3 → P1" in updated
    assert "Verified in *Action Items*" in updated
    assert not re.search(r"\b(?:U|F|Col)[A-Z0-9]{6,}\b", created + duplicate + updated)


def test_natural_named_updates_modify_existing_task_without_creating(slack):
    target = slack.add(
        "review deployment documentation", assignee="UP", priority="P2",
        due="2026-09-30")
    assert "updated" in ask("change the priority of review deployment documentation to P1")
    assert slack_tools.extract_priority(target, SCHEMA) == "P1"
    assert "updated" in ask(
        "change the due date of review deployment documentation to October 5")
    assert slack_tools.extract_due_date(target, SCHEMA) == "2026-10-05"
    assert "completed" in ask("mark review deployment documentation as completed")
    assert slack_tools.extract_completed(target, SCHEMA)
    assert len(slack.items) == 1
    assert all(not method.endswith("create") for method, _ in slack.writes)


def test_compact_displayed_row_can_safely_reference_its_task(slack):
    target = slack.add("Complete the testing report", assignee="UAA", priority="P1",
                       due="2026-10-03")
    response = ask("Complete the testing report · P1 · AasthaA · Oct 3")
    assert "Action item completed" in response
    assert slack_tools.extract_completed(target, SCHEMA)
    assert all(not method.endswith("create") for method, _ in slack.writes)


def test_ambiguous_task_response_contains_choices_but_not_internal_ids(slack):
    slack.add("Client report draft", assignee="UA")
    slack.add("Client report review", assignee="UM")
    response = ask("complete client report")
    assert "I found 2 tasks matching" in response
    assert "Which client report task" in response
    assert "Client report draft" in response and "Client report review" in response
    assert "ID:" not in response


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
        {"ok": True, "items": [{"id": "A"}, {"id": "B"}]}]
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
    assert "Proposed Changes" in response and not captured
    response = ask("confirm", thread="FILTERED_ASSIGN")
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
    assert "Status: Pending → Completed" in response
    assert "· Pending" not in response
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
    assert "Proposed Changes" in response and not slack.writes
    response = ask("confirm")
    assert "Action items completed" in response
    assert all(slack_tools.extract_completed(item, SCHEMA) for item in items)


def test_admin_bulk_update_and_reassignment_use_exact_collections(slack):
    items = [slack.add("Alpha", assignee="UM"), slack.add("Beta", assignee="UAA")]
    assert "Proposed Changes" in ask("change priority of all action items to P1")
    assert "Action items updated" in ask("confirm")
    assert all(slack_tools.extract_priority(item, SCHEMA) == "P1" for item in items)
    assert "Proposed Changes" in ask("assign all pending tasks to Morgan")
    assert "Action items updated" in ask("confirm")
    assert all(slack_tools.extract_assignee_ids(item, SCHEMA) == ["UM"] for item in items)


def test_member_cannot_delete_collection_owned_by_other_members(slack):
    slack.add("Admin work", assignee="UA")
    slack.add("Other work", assignee="UAA")
    response = ask("delete all action items", user="UM")
    assert "Permission denied" in response
    assert not slack.writes


def test_bulk_resolution_combines_member_status_priority_and_due_filters(slack):
    today = main.current_date()
    # Keep the fixture inside the current Monday-Sunday reporting week even
    # when the suite runs on Saturday or Sunday.
    in_week = (today + timedelta(days=min(2, 6 - today.weekday()))).isoformat()
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


def test_batch_member_resolution_uses_one_request_scoped_directory_snapshot(slack, monkeypatch):
    calls = []
    original = slack.users_list
    monkeypatch.setattr(slack, "users_list", lambda **kwargs: calls.append(kwargs) or original(**kwargs))
    command = commands.validate_command({
        "intent": "create", "operations": [], "changes": [],
        "tasks": [
            {"task_name": "Prepare report", "assignee": "Aastha"},
            {"task_name": "Review API", "assignee": "Praveen"},
        ],
    })
    resolved = main._resolve_command_members(
        command, main.context("UA", "C", "MEMBER-CACHE", None, "W"))
    assert [task["resolved_assignee_ids"] for task in resolved["tasks"]] == [["UAA"], ["UP"]]
    assert len(calls) == 1


def test_workspace_member_query_uses_slack_identity_and_dynamic_role(slack):
    parsed = commands.validate_command({"intent": "members", "members": ["Aastha"]})
    ctx = main.context("UA", "C", "T", None, "W")
    response = main._dispatch(main._resolve_command_members(parsed, ctx), ctx, "unused")
    assert "Aastha" in response and "UAA" not in response
    assert "viewer" in response


def test_model_unavailable_response_is_safe_and_confirms_no_list_change(slack):
    ctx = main.context("UA", "C", "T", None, "W")
    response = main._dispatch({"intent": "temporarily_unavailable"}, ctx, "unused")
    assert "AI extraction service is temporarily unavailable" in response
    assert "Slack List data was not changed" in response
    assert not slack.writes


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


def test_grouped_count_executes_after_member_and_status_filters(slack, monkeypatch):
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: {"UM": "Morgan", "UA": "Alex"}.get(user_id))
    slack.add("Morgan open", assignee="UM")
    slack.add("Morgan done", completed=True, assignee="UM")
    slack.add("Alex open", assignee="UA")
    response = ask("compare the number of open tasks assigned to Morgan and Alex")
    assert "Count by Assignee" in response
    assert "*Morgan*: 1" in response
    assert "*Alex*: 1" in response
    assert "*UM*" not in response and "*UA*" not in response
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

    assert "Proposed Changes" in ask("complete all action items", user="U_MANAGER")
    assert "Action items completed" in ask("confirm", user="U_MANAGER")
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
    assert "3 total" in response
    assert "1 completed" in response
    assert "2 pending" in response
    assert "1 overdue" in response
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
    assert "Metric" in response and "Count" in response
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
    assert re.search(rf"{today.isoformat()}\s+2", response)
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
    assert "Completion Unavailable" in rendered
    assert "At-risk tasks require" in rendered


def test_progress_distributions_and_deadlines_are_calculated_from_current_items(slack):
    today = main.current_date()
    within_week = today + timedelta(days=min(1, 6 - today.weekday()))
    slack.add("Done", completed=True, priority="P1", due=(today - timedelta(days=4)).isoformat())
    slack.add("Late", priority="P1", due=(today - timedelta(days=1)).isoformat())
    slack.add("Today", priority="P2", due=today.isoformat())
    slack.add("Later", priority="P3", due=within_week.isoformat())
    report = progress_engine.calculate_progress(
        slack.items, SCHEMA, today=today,
        metrics=["overview", "status_distribution", "priority_distribution", "due_today", "due_this_week"])
    assert report.snapshot == {
        "total": 4, "completed": 1, "pending": 3, "overdue": 1,
        "due_today": 2 if today.weekday() == 6 else 1,
        "due_this_week": 2, "completion_rate": 25.0,
    }
    assert report.status_distribution == {"Completed": 1, "Pending": 2, "Overdue": 1}
    assert report.priority_distribution == {"P1": 2, "P2": 1, "P3": 1}
    expected_due_today = [slack.items[2]["id"]]
    if today.weekday() == 6:
        expected_due_today.append(slack.items[3]["id"])
    assert [item["id"] for item in report.due_today_items] == expected_due_today


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


def _media_action(title="Review release", *, confidence=.95, assignee=None,
                  due_date=None, clarification=None, source_type="transcript",
                  priority="P2", operation="create"):
    return action_item_extraction.ExtractedAction(
        title, assignee, due_date, priority, "pending", source_type, "SOURCE-1",
        confidence, f"Evidence for {title}", clarification, operation)


def _mock_media(monkeypatch, actions, warnings=None):
    monkeypatch.setattr(content_ingestion, "ingest", lambda *args, **kwargs: (
        [content_ingestion.IngestedContent("transcript", "transcript", "SOURCE-1")], warnings or []))
    monkeypatch.setattr(action_item_extraction, "extract", lambda *args, **kwargs: list(actions))


def test_media_extraction_creates_multiple_items_through_verified_existing_pipeline(slack, monkeypatch):
    _mock_media(monkeypatch, [_media_action("Review release"), _media_action("Publish notes")])
    response = main.process_shared_content(
        "Create action items from this transcript", [], [], "UA", "C", "MEDIA", "1", "W")
    assert "created successfully" in response
    assert {slack_tools.extract_item_name(item, SCHEMA) for item in slack.items} == {
        "Review release", "Publish notes"}
    assert all(method.endswith("create") for method, _ in slack.writes)


def test_media_single_item_response_uses_one_classified_summary(slack, monkeypatch):
    _mock_media(monkeypatch, [_media_action("Prepare release brief", source_type="audio")])
    response = main.process_shared_content(
        "Extract action items from this audio", [], [], "UA", "C", "MEDIAONE", "1", "W")
    assert response.startswith("*🎧 Audio processed · 1 action item extracted*")
    assert response.count("Prepare release brief") == 1
    assert "*Created · 1*" in response


def test_explicit_media_update_reuses_verified_mutation_pipeline(slack, monkeypatch):
    task = slack.add("Review API documentation", assignee="UA", priority="P2")
    _mock_media(monkeypatch, [_media_action(
        "Review API documentation", priority="P1", operation="update")])
    response = main.process_shared_content(
        "Apply the updates from this transcript", [], [], "UA", "C", "MEDIAUPDATE", "1", "W")
    assert "*Updated · 1*" in response
    assert slack_tools.extract_priority(task, SCHEMA) == "P1"
    assert all(not method.endswith("create") for method, _ in slack.writes)


def test_media_extraction_reports_no_action_items_without_calling_it_an_empty_list(slack, monkeypatch):
    _mock_media(monkeypatch, [])
    response = main.process_shared_content(
        "Extract action items from this audio", [], [], "UA", "C", "NOITEMS", "1", "W")
    assert response.startswith("*📄 Transcript processed*")
    assert "couldn't identify any clear action items" in response
    assert "No action items found" not in response
    assert not slack.writes


def test_video_without_audio_returns_video_specific_error_without_extraction(slack, monkeypatch):
    calls = []
    monkeypatch.setattr(
        content_ingestion, "ingest",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            content_ingestion.ContentError("silent.mp4: The media contains no usable audio track.")))
    monkeypatch.setattr(action_item_extraction, "extract", lambda *args, **kwargs: calls.append(True))
    response = main.process_shared_content(
        "Extract action items from this video",
        [{"id": "FVIDEO", "name": "silent.mp4", "mimetype": "video/mp4"}], [],
        "UA", "C", "NOAUDIO", "1", "W")
    assert response == "*🎥 Video received, but no audio track was found.*"
    assert calls == [] and not slack.writes


def test_empty_audio_transcript_returns_safe_stage_specific_error(slack, monkeypatch):
    monkeypatch.setattr(
        content_ingestion, "ingest",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            content_ingestion.ContentError("meeting.mp3: The transcript is empty or contains no usable text.")))
    response = main.process_shared_content(
        "Extract action items from this audio",
        [{"id": "FAUDIO", "name": "meeting.mp3", "mimetype": "audio/mpeg"}], [],
        "UA", "C", "EMPTYAUDIO", "1", "W")
    assert "Audio transcription failed" in response
    assert "No usable speech" in response
    assert "FAUDIO" not in response and not slack.writes


def test_media_preview_then_confirmation_creates_exact_reviewed_items(slack, monkeypatch):
    action = _media_action("Validate migration", due_date=(main.current_date() + timedelta(days=2)).isoformat())
    _mock_media(monkeypatch, [action])
    preview = main.process_shared_content(
        "Extract action items but don't create them yet", [], [], "UA", "C", "PREVIEW", "1", "W")
    assert "No tasks were created" in preview
    assert "Validate migration" in preview and not slack.writes
    confirmed = ask("confirm", thread="PREVIEW")
    assert "created successfully" in confirmed
    assert slack_tools.extract_item_name(slack.items[0], SCHEMA) == "Validate migration"
    trace = source_trace.get(main.DB_PATH, "L", slack.items[0]["id"])
    assert trace["source_type"] == "transcript"
    assert "Evidence" in trace["evidence"]
    source_response = ask("Where did this task come from?", thread="PREVIEW")
    assert "Source for Validate migration" in source_response
    assert "Evidence" in source_response


def test_low_confidence_media_extraction_requires_confirmation(slack, monkeypatch):
    _mock_media(monkeypatch, [_media_action("Investigate option", confidence=.55)])
    response = main.process_shared_content(
        "Extract tasks from the recording", [], [], "UA", "C", "LOW", "1", "W")
    assert "Low-confidence" in response
    assert "⚠️ review" in response
    assert not slack.writes


def test_media_confirmation_is_invalidated_when_list_state_changes(slack, monkeypatch):
    _mock_media(monkeypatch, [_media_action("Review architecture", confidence=.6)])
    main.process_shared_content(
        "Extract tasks from recording", [], [], "UA", "C", "STALEMEDIA", "1", "W")
    slack.add("Concurrent task")
    response = ask("confirm", thread="STALEMEDIA")
    assert "changed after confirmation" in response
    assert not slack.writes


def test_ambiguous_media_reference_never_creates_or_stages_confirmation(slack, monkeypatch):
    _mock_media(monkeypatch, [_media_action(
        "Send document", confidence=.4, clarification="Which document and owner were intended?")])
    response = main.process_shared_content(
        "Extract tasks from this video", [], [], "UA", "C", "AMB", "1", "W")
    assert "Clarification required" in response
    assert "Which document" in response
    assert not slack.writes
    assert not main._state(main.context("UA", "C", "AMB", "1", "W")).get("confirmation")


def test_unknown_spoken_assignee_requires_member_clarification(slack, monkeypatch):
    _mock_media(monkeypatch, [_media_action(
        "Prepare client report", assignee="OSTHO", source_type="audio")])
    monkeypatch.setattr(slack_tools, "find_user_candidates", lambda value, members=None: [])
    response = main.process_shared_content(
        "Extract tasks from this audio", [], [], "UA", "C", "UNKNOWNMEDIA", "1", "W")
    assert "Member clarification required" in response
    assert "OSTHO" in response
    assert "@mention or display name" in response
    assert "No tasks were created" in response
    assert not slack.writes


def test_transcribed_austa_resolves_to_unique_aasthaa_member(slack, monkeypatch):
    slack.users[:] = [
        {"id": "UA", "name": "Alex"},
        {"id": "U0C2F3CFQ00", "name": "aasthaa", "real_name": "Aastha Acharya",
         "profile": {"display_name": "AasthaA", "real_name": "Aastha Acharya"}},
        {"id": "UP", "name": "Praveen"},
    ]
    due_a = (main.current_date() + timedelta(days=3)).isoformat()
    due_p = (main.current_date() + timedelta(days=5)).isoformat()
    _mock_media(monkeypatch, [
        _media_action("Prepare the client report", assignee="AUSTA", due_date=due_a,
                      source_type="audio"),
        _media_action("Review the API documentation", assignee="Praveen", due_date=due_p,
                      source_type="audio"),
        _media_action("Complete the deployment checklist", assignee="AUSTA",
                      due_date=(main.current_date() + timedelta(days=10)).isoformat(),
                      source_type="audio"),
    ])
    response = main.process_shared_content(
        "Extract tasks from this audio", [], [], "UA", "C", "AUSTA", "1", "W")
    assert "created successfully" in response
    assert "Member clarification required" not in response
    by_name = {slack_tools.extract_item_name(item, SCHEMA): item for item in slack.items}
    assert slack_tools.extract_assignee_ids(by_name["Prepare the client report"], SCHEMA) == [
        "U0C2F3CFQ00"]
    assert slack_tools.extract_assignee_ids(by_name["Review the API documentation"], SCHEMA) == ["UP"]
    assert slack_tools.extract_assignee_ids(by_name["Complete the deployment checklist"], SCHEMA) == [
        "U0C2F3CFQ00"]
    assert slack_tools.extract_due_date(by_name["Prepare the client report"], SCHEMA) == due_a
    assert slack_tools.extract_priority(by_name["Prepare the client report"], SCHEMA) == "P2"
    trace = source_trace.get(
        main.DB_PATH, "L", slack_tools.extract_item_id(by_name["Prepare the client report"]))
    assert trace["source_type"] == "audio"
    assert trace["evidence"] == "Evidence for Prepare the client report"


def test_unresolved_media_assignee_does_not_discard_resolvable_tasks(slack, monkeypatch):
    _mock_media(monkeypatch, [
        _media_action("Review the API documentation", assignee="Praveen", source_type="audio"),
        _media_action("Prepare unknown handoff", assignee="OSTHO", source_type="audio"),
    ])
    response = main.process_shared_content(
        "Extract tasks from this audio", [], [], "UA", "C", "PARTIALMEDIA", "1", "W")
    assert [slack_tools.extract_item_name(item, SCHEMA) for item in slack.items] == [
        "Review the API documentation"]
    assert slack_tools.extract_assignee_ids(slack.items[0], SCHEMA) == ["UP"]
    assert "Member clarification required" in response
    assert "Prepare unknown handoff" in response and "OSTHO" in response
    assert "resolved tasks were processed" in response
    assert "No tasks were created" not in response


def test_video_results_are_grouped_into_compact_outcomes(slack, monkeypatch):
    slack.add("Existing release", assignee="UA")
    _mock_media(monkeypatch, [
        _media_action("Prepare launch notes", assignee="Praveen", source_type="video"),
        _media_action("Existing release", assignee="Alex", source_type="video"),
        _media_action("Unknown handoff", assignee="OSTHO", source_type="video"),
    ])
    response = main.process_shared_content(
        "Extract tasks from this video", [], [], "UA", "C", "VIDEO-GROUPS", "1", "W")
    assert response.startswith("*🎥 Video processed · 3 action items extracted*")
    assert "*Created · 1*" in response
    assert "*Already exists · 1*" in response
    assert "*Needs clarification · 1*" in response
    assert "*Summary* · 1 created · 1 existing · 1 clarification" in response
    assert "Assignee:" not in response and "Priority:" not in response
    assert [slack_tools.extract_item_name(item, SCHEMA) for item in slack.items] == [
        "Existing release", "Prepare launch notes"]


def test_media_creation_enforces_existing_assignment_permissions(slack, monkeypatch):
    _mock_media(monkeypatch, [_media_action("Other-owned task", assignee="Alex")])
    response = main.process_shared_content(
        "Create tasks from transcript", [], [], "UM", "C", "RBACMEDIA", "1", "W")
    assert "only create tasks assigned to you" in response
    assert not slack.writes


def test_viewer_media_request_is_rejected_before_ingestion(slack, monkeypatch):
    called = []
    monkeypatch.setattr(content_ingestion, "ingest", lambda *args, **kwargs: called.append(True))
    response = main.process_shared_content(
        "Create tasks from transcript", [], [], "UV", "C", "VIEWMEDIA", "1", "W")
    assert "Permission denied" in response
    assert called == [] and not slack.writes


def test_media_duplicate_protection_reuses_existing_creation_guard(slack, monkeypatch):
    slack.add("Review release", assignee="UA")
    _mock_media(monkeypatch, [_media_action("Review release", assignee="Alex")])
    response = main.process_shared_content(
        "Create tasks from recording", [], [], "UA", "C", "DUPMEDIA", "1", "W")
    assert "Already exists" in response
    assert not slack.writes


def test_media_exact_duplicate_with_different_fields_preserves_requested_values(slack, monkeypatch, caplog):
    monkeypatch.setattr(main, "current_date", lambda: date(2026, 9, 24))
    slack.add("Review API documentation", assignee="UP", priority="P1", due="2026-10-25")
    requested = _media_action(
        "Review API documentation", assignee="<@UP>", due_date="2026-10-27")
    _mock_media(monkeypatch, [requested])
    with caplog.at_level("INFO", logger="slack_list"):
        response = main.process_shared_content(
            "Create tasks from transcript", [], [], "UA", "C", "MEDIAFIELDS", "1", "W")
    assert "Task already exists with different fields" in response
    assert "Oct 27" in response and "Oct 25" in response
    assert "No fields were changed" in response
    assert requested.due_date == "2026-10-27"
    assert slack_tools.extract_due_date(slack.items[0], SCHEMA) == "2026-10-25"
    assert slack_tools.extract_priority(slack.items[0], SCHEMA) == "P1"
    assert not slack.writes
    assert "Duplicate comparison input" in caplog.text
    assert "due_date=2026-10-27" in caplog.text


def test_media_creation_reports_verification_failure(slack, monkeypatch):
    slack.noop = True
    _mock_media(monkeypatch, [_media_action("Verify write")])
    response = main.process_shared_content(
        "Create tasks from recording", [], [], "UA", "C", "VERIFYMEDIA", "1", "W")
    assert "not verified" in response or "outcome unknown" in response


def test_file_share_event_passes_file_metadata_to_delivery(slack, monkeypatch):
    observed = {}
    monkeypatch.setattr(main, "_deliver", lambda *args, **kwargs: observed.update(kwargs))
    main.message_handler(
        {"team_id": "W"},
        {"type": "message", "subtype": "file_share", "user": "UA", "channel": "D1",
         "ts": "1", "text": "extract tasks", "files": [{"id": "F1", "mimetype": "audio/ogg"}]}, {})
    assert observed["files"][0]["id"] == "F1"


def test_plain_text_create_routes_only_to_text_pipeline(slack, monkeypatch, caplog):
    command = (
        "create a task to review the deployment documentation for Praveen "
        "by September 30 with priority P1"
    )
    replies = []
    monkeypatch.setattr(
        main, "process_shared_content",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("plain text entered shared-content processing")))
    monkeypatch.setattr(
        content_ingestion, "ingest",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("plain text invoked transcription ingestion")))
    monkeypatch.setattr(
        main, "send_and_record",
        lambda channel, text, *args, **kwargs: replies.append(text) or {"ts": "TEXT-REPLY"})

    with caplog.at_level("INFO"):
        main._deliver("PLAIN-TEXT-ROUTE", command, "UA", "C", "TEXT-THREAD", "1", "W")

    assert len(slack.items) == 1
    item = slack.items[0]
    assert slack_tools.extract_item_name(item, SCHEMA) == "review the deployment documentation"
    assert slack_tools.extract_assignee_ids(item, SCHEMA) == ["UP"]
    assert slack_tools.extract_due_date(item, SCHEMA) == "2026-09-30"
    assert slack_tools.extract_priority(item, SCHEMA) == "P1"
    assert any(method.endswith("create") for method, _ in slack.writes)
    assert replies and "Task created" in replies[0]
    assert "request_routed source=text route=text" in caplog.text
    assert "intent=create" in caplog.text
    assert "llm_used=false llm_call_count=0" in caplog.text


@pytest.mark.parametrize("kind,mimetype", [("audio", "audio/mpeg"), ("video", "video/mp4")])
def test_attached_media_routes_through_transcription_and_extraction(
        slack, monkeypatch, caplog, kind, mimetype):
    observed = {"transcription": [], "extraction": 0}
    original_ingest = content_ingestion.ingest

    def transcriber(data, media_kind, mime):
        observed["transcription"].append((media_kind, mime, data))
        return content_ingestion.transcription.Transcript(
            f"Prepare the {kind} routing report.", 1, 1.0)

    def ingest(text, files, attachments, token, slack_client=None):
        return original_ingest(
            text, files, attachments, token,
            downloader=lambda source, bot_token: b"media-bytes",
            transcriber=transcriber,
            slack_client=slack_client,
            metadata_resolver=lambda source, client: source,
        )

    def extract(contents, today):
        observed["extraction"] += 1
        return [_media_action(f"Prepare the {kind} routing report", source_type=kind)]

    monkeypatch.setattr(content_ingestion, "ingest", ingest)
    monkeypatch.setattr(action_item_extraction, "extract", extract)
    monkeypatch.setattr(
        main, "process",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("attached media entered text processing")))
    monkeypatch.setattr(main, "post", lambda *args, **kwargs: {"ts": "MEDIA-STATUS"})
    monkeypatch.setattr(main, "update_and_record", lambda *args, **kwargs: {"ts": "MEDIA-STATUS"})

    files = [{"id": f"F-{kind}", "mimetype": mimetype, "name": f"sample.{kind}"}]
    with caplog.at_level("INFO"):
        main._deliver(
            f"{kind.upper()}-ROUTE", f"extract tasks from this {kind}", "UA", "C",
            f"{kind.upper()}-THREAD", "1", "W", files=files)

    assert observed["transcription"] == [(kind, mimetype, b"media-bytes")]
    assert observed["extraction"] == 1
    assert slack_tools.extract_item_name(slack.items[0], SCHEMA) == \
        f"Prepare the {kind} routing report"
    assert f"request_routed source={kind} route=media" in caplog.text


def test_media_delivery_posts_one_idempotent_processing_status(slack, monkeypatch):
    posted, delivered, updated = [], [], []
    monkeypatch.setattr(content_ingestion, "should_ingest", lambda *args: True)
    monkeypatch.setattr(main, "process_shared_content", lambda *args, **kwargs: "final response")
    monkeypatch.setattr(
        main, "post",
        lambda channel, text, thread_ts=None, metadata=None:
        posted.append((channel, text, thread_ts)) or {"ts": "STATUS"})
    monkeypatch.setattr(
        main, "send_and_record",
        lambda channel, text, *args, **kwargs:
        delivered.append((channel, text)) or {"ts": "FINAL"})
    monkeypatch.setattr(main, "app", SimpleNamespace(client=SimpleNamespace(
        chat_update=lambda **kwargs: updated.append(kwargs) or {"ts": kwargs["ts"]})))
    files = [{"id": "FVIDEO", "name": "meeting.mp4", "mimetype": "video/mp4"}]
    main._deliver("MEDIA-STATUS", "extract tasks", "UA", "C", "THREAD", "1", "W", files=files)
    main._deliver("MEDIA-STATUS", "extract tasks", "UA", "C", "THREAD", "1", "W", files=files)
    assert posted == [("C", "*🎥 Processing video…*", "THREAD")]
    assert delivered == []
    assert updated == [{"channel": "C", "ts": "STATUS", "text": "final response"}]


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
    assert "1 total" in response
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
    within_week = today + timedelta(days=min(1, 6 - today.weekday()))
    first = slack.add("Urgent plan item", assignee="UA", priority="P1",
                      due=within_week.isoformat())
    second = slack.add("Later plan item", assignee="UA", priority="P3",
                       due=within_week.isoformat())
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


def test_smart_search_by_keyword_is_case_insensitive_and_read_only(slack):
    wanted = slack.add("Review Deployment Documentation", assignee="UP")
    slack.add("Prepare client report", assignee="UP")
    before = deepcopy(slack.items)
    response = ask("find deployment tasks", thread="SEARCH_KEYWORD")
    assert "Task Search" in response and "Review Deployment Documentation" in response
    assert "Prepare client report" not in response
    assert "1 matching task" in response
    assert slack.items == before and not slack.writes
    state = main._state(main.context("UA", "C", "SEARCH_KEYWORD", None, "W"))
    assert [entry["item_id"] for entry in state["displayed_tasks"]] == [wanted["id"]]


def test_smart_search_by_multiple_keywords(slack):
    slack.add("Prepare quarterly client report", assignee="UP")
    slack.add("Prepare internal report", assignee="UP")
    response = ask("search client report", thread="SEARCH_WORDS")
    assert "quarterly client report" in response and "internal report" not in response


def test_smart_search_by_assignee(slack):
    slack.add("Praveen urgent", assignee="UP", priority="P1")
    slack.add("Aastha urgent", assignee="UAA", priority="P1")
    response = ask("find Praveen's P1 tasks", thread="SEARCH_OWNER")
    assert "Task Search" in response and "Praveen urgent" in response
    assert "Aastha urgent" not in response


def test_smart_search_by_priority(slack):
    slack.add("Critical deployment", assignee="UP", priority="P1")
    slack.add("Normal deployment", assignee="UP", priority="P3")
    response = ask("find P1 deployment tasks", thread="SEARCH_PRIORITY")
    assert "Critical deployment" in response and "Normal deployment" not in response


def test_smart_search_by_status(slack):
    slack.add("Deployment open", assignee="UP")
    slack.add("Deployment archived", completed=True, assignee="UP")
    response = ask("search completed deployment tasks", thread="SEARCH_STATUS")
    assert "Deployment archived" in response and "Deployment open" not in response


def test_smart_search_by_due_date(slack):
    today = main.current_date()
    in_week = today + timedelta(days=max(0, 6 - today.weekday()))
    later = in_week + timedelta(days=7)
    slack.add("This week", assignee="UP", due=in_week.isoformat())
    slack.add("Later", assignee="UP", due=later.isoformat())
    response = ask("search tasks due this week", thread="SEARCH_DUE")
    assert "This week" in response and "Later" not in response


def test_smart_search_combines_keyword_owner_priority_status_and_due(slack):
    today = main.current_date()
    in_week = today + timedelta(days=max(0, 6 - today.weekday()))
    slack.add("Deployment release", assignee="UP", priority="P1", due=in_week.isoformat())
    slack.add("Deployment low", assignee="UP", priority="P3", due=in_week.isoformat())
    slack.add("Deployment mine", assignee="UA", priority="P1", due=in_week.isoformat())
    response = ask(
        "find Praveen's pending P1 deployment tasks due this week", thread="SEARCH_COMBINED")
    assert "Deployment release" in response
    assert "Deployment low" not in response and "Deployment mine" not in response


def test_smart_search_zero_results_is_clean_and_never_mutates(slack):
    slack.add("Client report", assignee="UP")
    before = deepcopy(slack.items)
    response = ask("find nonexistent deployment tasks", thread="SEARCH_EMPTY")
    assert response == "*🔎 Task Search*\n\nNo matching action items found."
    assert slack.items == before and not slack.writes


@pytest.mark.parametrize("wording,field,value", [
    ("Move all my overdue tasks to next Monday", "due_date", None),
    ("Change all my P3 deployment tasks to P2", "priority", "P2"),
    ("Assign all unassigned deployment tasks to me", "assignee", "me"),
    ("Move my tasks due this week to Friday", "due_date", None),
    ("Complete all my testing tasks", "completed", True),
])
def test_bulk_proposal_language_is_parsed_locally(wording, field, value):
    parsed = intent_parser.parse_intent(wording)
    assert parsed["intent"] in {"update", "complete"}
    assert parsed["bulk_preview_required"] is True
    if parsed["intent"] == "complete":
        assert field == "completed"
    else:
        change = parsed["changes"][0]
        assert change["field"] == field
        if value is not None:
            assert change["value"] == value


def test_bulk_operation_previews_then_confirms_exact_changes(slack):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    first = slack.add("Deployment one", assignee="UA", due=yesterday)
    second = slack.add("Deployment two", assignee="UA", due=yesterday)
    preview = ask("Move all my overdue tasks to next Monday", thread="BULK_PREVIEW")
    assert "Proposed Changes" in preview and "2 tasks will be updated" in preview
    assert "Due:" in preview and "Owner" not in preview
    assert not slack.writes
    response = ask("do it", thread="BULK_PREVIEW")
    assert "Action items updated" in response
    assert all(slack_tools.extract_due_date(item, SCHEMA) > main.current_date().isoformat()
               for item in (first, second))


def test_bulk_operation_cancellation_never_writes(slack):
    slack.add("Testing one", assignee="UA")
    slack.add("Testing two", assignee="UA")
    assert "Proposed Changes" in ask("Complete all my testing tasks", thread="BULK_CANCEL_NEW")
    assert "Cancelled" in ask("don't", thread="BULK_CANCEL_NEW")
    assert not slack.writes


def test_bulk_operation_rbac_denial_happens_before_preview(slack):
    slack.add("Testing one", assignee="UM")
    slack.add("Testing two", assignee="UM")
    response = ask("Complete all my testing tasks", user="UM", thread="BULK_RBAC_NEW")
    assert "Permission denied" in response and "bulk" in response.casefold()
    assert not slack.writes


def test_bulk_preview_detects_deleted_task_before_confirmation(slack):
    first = slack.add("Deployment one", assignee="UA", priority="P3")
    second = slack.add("Deployment two", assignee="UA", priority="P3")
    assert "Proposed Changes" in ask(
        "Change all my P3 deployment tasks to P2", thread="BULK_DELETED")
    slack.items.remove(second)
    response = ask("yes", thread="BULK_DELETED")
    assert "Some tasks changed since the preview" in response
    assert slack_tools.extract_priority(first, SCHEMA) == "P3" and not slack.writes


def test_bulk_preview_detects_stale_field_change(slack):
    first = slack.add("Deployment one", assignee="UA", priority="P3")
    slack.add("Deployment two", assignee="UA", priority="P3")
    assert "Proposed Changes" in ask(
        "Change all my P3 deployment tasks to P2", thread="BULK_STALE_NEW")
    first["fields"].append({"column_id": "external", "text": "changed elsewhere"})
    response = ask("apply", thread="BULK_STALE_NEW")
    assert "Some tasks changed since the preview" in response and not slack.writes


def test_bulk_partial_failure_reports_each_result(slack, monkeypatch):
    first = slack.add("Deployment one", assignee="UA", priority="P3")
    second = slack.add("Deployment two", assignee="UA", priority="P3")
    assert "Proposed Changes" in ask(
        "Change all my P3 deployment tasks to P2", thread="BULK_PARTIAL")
    original = slack_tools.update_action_item_field

    def fail_second(item_id, field, value, ctx, list_id):
        if item_id == second["id"]:
            raise RuntimeError("simulated Slack failure")
        return original(item_id, field, value, ctx, list_id)

    monkeypatch.setattr(slack_tools, "update_action_item_field", fail_second)
    response = ask("confirm", thread="BULK_PARTIAL")
    assert "not all changes verified" in response
    assert slack_tools.extract_priority(first, SCHEMA) == "P2"
    assert slack_tools.extract_priority(second, SCHEMA) == "P3"


def test_bulk_zero_matches_never_stages_confirmation(slack):
    slack.add("Client report", assignee="UA", priority="P3")
    response = ask("Change all my P3 deployment tasks to P2", thread="BULK_ZERO")
    assert "No Slack List items match" in response
    assert not main._state(main.context("UA", "C", "BULK_ZERO", None, "W")).get("confirmation")
    assert not slack.writes


def test_bulk_mixed_authorization_excludes_unsafe_tasks(slack, monkeypatch):
    own = slack.add("Testing mine", assignee="UM")
    other = slack.add("Testing other", assignee="UA")
    monkeypatch.setitem(config.PERMISSIONS, "member", config.PERMISSIONS["member"] | {"bulk"})
    preview = ask("complete all testing tasks", user="UM", thread="BULK_MIXED")
    assert "Proposed Changes" in preview and "Excluded" in preview
    assert "Testing mine" in preview and "Testing other" in preview
    response = ask("confirm", user="UM", thread="BULK_MIXED")
    assert "completed" in response
    assert slack_tools.extract_completed(own, SCHEMA)
    assert not slack_tools.extract_completed(other, SCHEMA)


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


def test_apply_plan_refetches_authorizes_updates_and_verifies(slack, monkeypatch):
    today = date(2026, 9, 23)
    monkeypatch.setattr(main, "current_date", lambda: today)
    first = slack.add("Plan one", assignee="UA", priority="P1", due=(today + timedelta(days=3)).isoformat())
    second = slack.add("Plan two", assignee="UA", priority="P2", due=(today + timedelta(days=4)).isoformat())
    ask("Plan my work for next week", thread="PLAN_APPLY")
    response = ask("apply this plan", thread="PLAN_APPLY")
    assert "Proposal applied and verified" in response
    assert slack_tools.extract_item_id(first) in {entry["id"] for entry in slack.items}
    assert slack_tools.extract_item_id(second) in {entry["id"] for entry in slack.items}
    assert len(slack.writes) == 2


def test_plan_application_fails_closed_when_target_changes(slack):
    today = main.current_date()
    item = slack.add("Plan stale", assignee="UA", priority="P1", due=today.isoformat())
    ask("Plan my work for this week", thread="PLAN_STALE")
    item["fields"].append({"column_id": "external", "text": "changed"})
    response = ask("apply this plan", thread="PLAN_STALE")
    assert "changed or no longer exist" in response
    assert not slack.writes


def test_verified_mutation_is_available_in_audit_history(slack, monkeypatch):
    monkeypatch.setattr(slack_tools, "user_display_name", lambda user_id: "Alex" if user_id == "UA" else None)
    item = slack.add("Audited", assignee="UA")
    ask("complete Audited", thread="AUDIT")
    response = ask("Who changed this task?", thread="AUDIT")
    assert "Verified mutation history" in response
    assert "complete" in response
    assert "Audited" in response
    assert item["id"] not in response
    assert "UA" not in response


def test_task_creation_is_recorded_in_persistent_history(slack):
    assert "created successfully" in ask(
        "create a task called History creation for me", thread="HISTORY_CREATE")
    response = ask("Show the history of the History creation task", thread="HISTORY_CREATE")
    assert "Task History" in response and "Created" in response
    assert "History creation" in response


def test_history_reports_assignment_priority_due_and_completion_changes(slack, monkeypatch):
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: {"UA": "AasthaA", "UP": "Praveen"}.get(user_id))
    due = (main.current_date() + timedelta(days=3)).isoformat()
    item = slack.add("History deployment", assignee="UA", priority="P3",
                     due=(main.current_date() + timedelta(days=1)).isoformat())
    assert "updated" in ask("assign History deployment to Praveen", thread="HISTORY_ALL")
    assert "updated" in ask("change the priority of History deployment to P1", thread="HISTORY_ALL")
    assert "updated" in ask(f"change the due date of History deployment to {due}", thread="HISTORY_ALL")
    assert "completed" in ask("complete History deployment", thread="HISTORY_ALL")
    response = ask("Show history of the History deployment task", thread="HISTORY_ALL")
    assert "Reassigned" in response and "Aastha" in response and "Praveen" in response
    assert "Priority: P3 → P1" in response
    assert "Due:" in response and "Completed" in response
    assert item["id"] not in response


def test_previous_due_date_uses_verified_history_only(slack):
    old_due = (main.current_date() + timedelta(days=1)).isoformat()
    new_due = (main.current_date() + timedelta(days=4)).isoformat()
    slack.add("Previous deadline", assignee="UA", due=old_due)
    ask(f"change the due date of Previous deadline to {new_due}", thread="HISTORY_PREVIOUS")
    response = ask("What was the previous due date?", thread="HISTORY_PREVIOUS")
    assert "Previous Due" in response
    assert slack_presentation.compact_date(old_due, main.current_date()) in response


def test_when_assigned_to_member_uses_actual_audit_event(slack, monkeypatch):
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: {"UA": "AasthaA", "UP": "Praveen"}.get(user_id))
    slack.add("Assignment history", assignee="UA")
    ask("assign Assignment history to Praveen", thread="HISTORY_ASSIGNED")
    response = ask("When was this task assigned to Praveen?", thread="HISTORY_ASSIGNED")
    assert "Owner history" in response and "Praveen" in response


def test_missing_history_discloses_tracking_boundary(slack):
    slack.add("Legacy task", assignee="UA")
    response = ask("Show history of the Legacy task", thread="HISTORY_MISSING")
    assert "No verified mutation history" in response
    assert "History tracking started" in response


def test_what_changed_today_filters_persistent_history(slack, monkeypatch):
    old = slack.add("Old history", assignee="UA")
    recent = slack.add("Recent history", assignee="UA")
    ctx = main.context("UA", "C", "HISTORY_DATE", None, "W")
    start = datetime.combine(main.current_date(), datetime.min.time(), main.ZoneInfo("Asia/Kathmandu"))
    monkeypatch.setattr(audit_log.time, "time", lambda: (start - timedelta(days=2)).timestamp())
    audit_log.record(main.DB_PATH, ctx, old["id"], "create", [], SCHEMA, None, old)
    monkeypatch.setattr(audit_log.time, "time", lambda: (start + timedelta(hours=9)).timestamp())
    audit_log.record(main.DB_PATH, ctx, recent["id"], "create", [], SCHEMA, None, recent)
    response = ask("What changed today?", thread="HISTORY_DATE")
    assert "Changes Today" in response and "Recent history" in response
    assert "Old history" not in response


def test_what_changed_today_reports_task_that_became_overdue(slack):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    slack.add("Website update", assignee="UA", due=yesterday)
    response = ask("What changed today?", thread="HISTORY_OVERDUE")
    assert "Became overdue" in response and "Website update" in response


def test_text_message_and_thread_sources_are_persisted(slack):
    main.process("create a task called Message context for me", "UA", "C", None, "M1", "W")
    message = ask("Show context for the Message context task", thread="SOURCE_MESSAGE")
    assert "Task Context" in message and "Slack message" in message
    ask("create a task called Thread context for me", thread="SOURCE_THREAD")
    thread = ask("Why does the Thread context task exist?", thread="SOURCE_THREAD")
    assert "Task Context" in thread and "Slack conversation" in thread


@pytest.mark.parametrize("wording,expected,reference_kind", [
    ("why does the Review the API documentation task exist?",
     "Review the API documentation", None),
    ("show context for Review the API documentation",
     "Review the API documentation", None),
    ("where did this task come from?", "__LAST__", "focus"),
])
def test_context_intent_extracts_only_the_task_reference(wording, expected, reference_kind):
    parsed = intent_parser.parse_intent(wording)
    assert parsed["intent"] == "source"
    assert parsed["task_name"] == expected
    assert (parsed.get("reference") or {}).get("kind") == reference_kind


def _add_ambiguous_context_tasks(slack, thread):
    first = slack.add(
        "Review the API documentation", assignee="UP", priority="P1",
        due=(main.current_date() + timedelta(days=2)).isoformat())
    second = slack.add("Review API documentation", priority="P3")
    ctx = main.context("UA", "C", thread, None, "W")
    source_trace.record(main.DB_PATH, ctx, first["id"], {
        "type": "slack_message", "evidence": "First task source",
    })
    source_trace.record(main.DB_PATH, ctx, second["id"], {
        "type": "transcript", "evidence": "Second task source",
    })
    return first, second


def test_ambiguous_context_response_is_compact_and_grammatical(slack, monkeypatch):
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: {"UA": "AasthaA", "UP": "Praveen"}.get(user_id))
    _add_ambiguous_context_tasks(slack, "SOURCE_AMBIGUOUS")
    response = ask(
        "show context for API documentation",
        thread="SOURCE_AMBIGUOUS")
    assert "I found 2 similar tasks. Which one do you mean?" in response
    assert "1. *Review the API documentation*" in response
    assert "P1 · Praveen" in response
    assert "2. *Review API documentation*" in response
    assert "P3 · Unassigned · No due date · Pending" in response
    assert "task task" not in response


@pytest.mark.parametrize("selection,evidence", [
    ("1", "First task source"),
    ("the first one", "First task source"),
    ("Review API documentation", "Second task source"),
])
def test_context_disambiguation_preserves_original_request(slack, monkeypatch, selection, evidence):
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: {"UA": "AasthaA", "UP": "Praveen"}.get(user_id))
    thread = "SOURCE_SELECT_" + re.sub(r"\W+", "_", selection)
    _add_ambiguous_context_tasks(slack, thread)
    assert "Which one do you mean?" in ask(
        "show context for API documentation", thread=thread)
    response = ask(selection, thread=thread)
    assert "Task Context" in response and evidence in response
    assert "Priority:" in response


@pytest.mark.parametrize("wording", [
    "weekly summary", "show weekly summary", "weekly action item summary",
    "summarize this week", "give me this week's summary",
    "summarize the week's action items", "show this week's action items summary",
    "weekly action items report",
])
def test_weekly_summary_variants_are_deterministic(wording, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("weekly summary should not call the language model")

    monkeypatch.setattr(intent_parser, "_configured_ollama_client", forbidden)
    assert intent_parser.parse_intent(wording)["intent"] == "weekly_summary"


def test_weekly_period_is_local_monday_through_sunday():
    assert main.weekly_period(date(2026, 9, 23)) == (
        date(2026, 9, 21), date(2026, 9, 27))


def test_weekly_summary_counts_verified_activity_and_current_state(slack, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "weekly.sqlite3"))
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: {"UA": "AasthaA", "UP": "Praveen"}.get(user_id))
    today = main.current_date()
    created = slack.add("Created this week", assignee="UP", priority="P1")
    completed = slack.add("Completed this week", completed=True, assignee="UA", priority="P2")
    overdue = slack.add(
        "Overdue this week", assignee="UP", priority="P1",
        due=(today - timedelta(days=1)).isoformat())
    slack.add("Ordinary pending", assignee="UA", priority="P3")
    ctx = main.context("UA", "C", "WEEKLY", None, "W")
    start, _ = main.weekly_period(today)
    moment = datetime.combine(start + timedelta(days=1), datetime.min.time(),
                              main.ZoneInfo("Asia/Kathmandu")).timestamp()
    monkeypatch.setattr(audit_log.time, "time", lambda: moment)
    audit_log.record(main.DB_PATH, ctx, created["id"], "create", [], SCHEMA, None, created)
    before = deepcopy(completed)
    next(field for field in before["fields"] if field["column_id"] == "done")["checkbox"] = False
    audit_log.record(
        main.DB_PATH, ctx, completed["id"], "complete",
        [{"field": "completed", "value": True}], SCHEMA, before, completed)

    response = ask("weekly summary", thread="WEEKLY")
    assert "Weekly Action Items Summary" in response
    assert "Created:* 1" in response and "Completed:* 1" in response
    assert "*Pending · 3*" in response and "Overdue:* 1" in response
    assert "P1 2" in response and "P3 1" in response
    assert "*Pending · 3*" in response
    assert "Created this week" in response and "Ordinary pending" in response
    assert "Task" in response and "Priority" in response and "Owner" in response and "Due" in response
    assert "Overdue this week — Praveen — P1" in response
    assert "Completed this week" in response
    assert overdue["id"] not in response
    assert "UP" not in response and "UA" not in response


def test_weekly_summary_empty_state_is_clean(slack, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "empty-week.sqlite3"))
    response = ask("show weekly summary", thread="WEEKLY_EMPTY")
    assert "Created:* 0" in response and "*Pending · 0*" in response
    assert "*Pending · 0*" in response
    assert "No pending action items this week." in response


def test_weekly_summary_respects_requester_rbac(slack, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "weekly-rbac.sqlite3"))
    own = slack.add("Member visible", assignee="UM", priority="P2")
    other = slack.add("Admin private", assignee="UA", priority="P1")
    response = ask("summarize this week", user="UM", thread="WEEKLY_RBAC")
    assert "*Pending · 1*" in response
    assert "Admin private" not in response and other["id"] not in response
    assert own["id"] not in response


def test_follow_up_loader_targets_owner_and_uses_authorized_unassigned_fallback(
        slack, monkeypatch, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(config, "DEFAULT_LIST_ID", "L")
    monkeypatch.setattr(config, "SLACK_LIST_CHANNEL_ID", "C")
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: {"UA": "AasthaA", "UP": "Praveen"}.get(user_id))
    today = main.current_date().isoformat()
    next(user for user in slack.users if user["id"] == "UP")["tz"] = "America/New_York"
    next(user for user in slack.users if user["id"] == "UAA")["tz"] = "Asia/Kathmandu"
    slack_tools.user_timezone.cache_clear()
    slack.add("Praveen deadline", assignee="UP", due=today)
    slack.add("Aastha deadline", assignee="UA", due=today)
    slack.add("AasthaA Kathmandu deadline", assignee="UAA", due=today)
    slack.add("Unassigned deadline", due=today)
    slack.add("Completed deadline", completed=True, assignee="UP", due=today)
    settings = deadline_reminders.ReminderSettings(
        fallback_channel="C2", fallback_actor_id="UA",
        workspace_timezone="Europe/London", timezone="UTC")
    tasks = main._load_deadline_reminder_tasks(settings)
    by_name = {task.name: task for task in tasks}
    assert by_name["Praveen deadline"].owner_id == "UP"
    assert by_name["Praveen deadline"].owner_name == "Praveen"
    assert by_name["Praveen deadline"].timezone == "Europe/London"
    assert by_name["Aastha deadline"].timezone == "Europe/London"
    assert by_name["AasthaA Kathmandu deadline"].timezone == "Europe/London"
    assert by_name["Unassigned deadline"].owner_id == "channel:C2"
    assert "Completed deadline" not in by_name
    assert "source=team_timezone" in caplog.text


def test_follow_up_timezone_uses_slack_profile_when_team_timezone_missing(
        slack, monkeypatch, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(config, "DEFAULT_LIST_ID", "L")
    monkeypatch.setattr(config, "SLACK_LIST_CHANNEL_ID", "C")
    next(user for user in slack.users if user["id"] == "UP")["tz"] = "America/New_York"
    slack_tools.user_timezone.cache_clear()
    slack.add("Profile timezone", assignee="UP", due=main.current_date().isoformat())
    tasks = main._load_deadline_reminder_tasks(
        deadline_reminders.ReminderSettings(workspace_timezone="", timezone="UTC"))
    assert tasks[0].timezone == "America/New_York"
    assert "source=slack_profile" in caplog.text


def test_follow_up_timezone_uses_application_default_when_profile_and_workspace_missing(
        slack, monkeypatch, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(config, "DEFAULT_LIST_ID", "L")
    monkeypatch.setattr(config, "SLACK_LIST_CHANNEL_ID", "C")
    slack.add("Fallback timezone", assignee="UA", due=main.current_date().isoformat())
    slack_tools.user_timezone.cache_clear()
    settings = deadline_reminders.ReminderSettings(
        timezone="Europe/Paris", workspace_timezone="",
        application_timezone_configured=True)
    tasks = main._load_deadline_reminder_tasks(settings)
    assert tasks[0].timezone == "Europe/Paris"
    assert "source=application_fallback" in caplog.text


def test_follow_up_timezone_final_fallback_is_explicit(monkeypatch, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(slack_tools, "user_timezone", lambda user_id: None)
    timezone, source = main.resolve_follow_up_timezone(
        "UA", deadline_reminders.ReminderSettings(
            timezone="Asia/Kathmandu", workspace_timezone="",
            application_timezone_configured=False))
    assert timezone == "Asia/Kathmandu" and source == "default_fallback"
    assert "source=default_fallback" in caplog.text


@pytest.mark.parametrize("source_type,label", [
    ("audio", "Audio"), ("video", "Video"), ("transcript", "Transcript"),
])
def test_media_source_context_is_reported_without_fabrication(slack, source_type, label):
    item = slack.add(f"{label} context", assignee="UA")
    ctx = main.context("UA", "C", f"SOURCE_{label}", None, "W")
    source_trace.record(main.DB_PATH, ctx, item["id"], {
        "type": source_type, "evidence": f"Verified {source_type} evidence",
    })
    response = ask(f"Show context for the {label} context task", thread=f"SOURCE_{label}")
    assert "Task Context" in response and f"Source:* {label}" in response
    assert f"Verified {source_type} evidence" in response


def test_missing_task_context_is_explicit(slack):
    slack.add("No context", assignee="UA")
    response = ask("Show context for the No context task", thread="SOURCE_NONE")
    assert "No source context is available" in response


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


def test_bulk_confirmation_is_scoped_to_requesting_user(slack):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    slack.add("Private overdue one", assignee="UA", due=yesterday)
    slack.add("Private overdue two", assignee="UA", due=yesterday)
    assert "Proposed Changes" in ask(
        "move all my overdue tasks to next Monday", user="UA", thread="BULK_OWNER")
    assert "no current confirmation" in ask(
        "confirm", user="UM", thread="BULK_OWNER")
    assert not slack.writes


def test_visualization_renderer_consumes_structured_report_only():
    import visualization
    report = progress_engine.ProgressReport(
        requested=("priority_distribution",),
        priority_distribution={"P1": 2, "P2": 1})
    rendered = visualization.render_progress(report, lambda items, title: title)
    assert "Priority distribution" in rendered
    assert "P1" in rendered and "Metric" in rendered and "Count" in rendered


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


@pytest.mark.parametrize("wording,intent", [
    ("what should I focus on?", "list"),
    ("what needs my attention today?", "list"),
    ("what should I work on first?", "list"),
    ("show risky tasks", "health"),
    ("what tasks are at risk?", "health"),
    ("how is the team workload?", "workload"),
    ("who has the most pending work?", "workload"),
    ("health", "health"),
    ("how are our action items?", "health"),
])
def test_task_intelligence_commands_are_deterministic(wording, intent, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("task intelligence should not call the language model")

    monkeypatch.setattr(intent_parser, "_configured_ollama_client", forbidden)
    parsed = commands.validate_command(intent_parser.parse_intent(wording))
    assert parsed["intent"] == intent


def test_daily_focus_orders_deadlines_and_priorities_and_excludes_completed(slack, monkeypatch):
    today = main.current_date()
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: {"UA": "AasthaA", "UP": "Praveen"}.get(user_id))
    slack.add("Due today P2", assignee="UA", priority="P2", due=today.isoformat())
    slack.add("Overdue P3", assignee="UA", priority="P3", due=(today - timedelta(days=1)).isoformat())
    slack.add("Overdue P1", assignee="UA", priority="P1", due=(today - timedelta(days=1)).isoformat())
    slack.add("Due today P1", assignee="UA", priority="P1", due=today.isoformat())
    slack.add("Completed P1", completed=True, assignee="UA", priority="P1", due=today.isoformat())
    response = ask("what should I focus on today?", thread="INTELLIGENCE_FOCUS")
    assert response.index("Overdue P1") < response.index("Overdue P3")
    assert response.index("Overdue P3") < response.index("Due today P1")
    assert response.index("Due today P1") < response.index("Due today P2")
    assert "```" in response and "Completed P1" not in response
    assert not slack.writes


def test_daily_focus_distinguishes_no_due_today_from_no_pending(slack):
    slack.add("Future work", assignee="UA", priority="P1",
              due=(main.current_date() + timedelta(days=3)).isoformat())
    response = ask("what should I focus on?", thread="INTELLIGENCE_FUTURE")
    assert "No action items are due today" in response
    assert "No action items are due today.\n\n*Pending work:* 1 task" in response
    assert "*📅 Upcoming*" in response and "Future work" in response
    slack.items.clear()
    empty = ask("what should I focus on?", thread="INTELLIGENCE_EMPTY")
    assert "No pending action items are assigned to you" in empty


@pytest.mark.parametrize("phrase", [
    "what should I focus on this week?",
    "what should I focus on for the week?",
    "what are my priorities this week?",
    "what tasks should I focus on this week?",
    "what should I work on this week?",
])
def test_weekly_focus_variations_route_deterministically_without_llm(phrase, monkeypatch):
    monkeypatch.setattr(
        intent_parser, "_configured_ollama_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("weekly focus should be deterministic")))
    parsed = commands.validate_command(intent_parser.parse_intent(phrase))
    assert parsed["intent"] == "weekly_focus"
    assert parsed["assignee_self"] is True


def test_today_focus_route_remains_distinct_from_weekly_focus():
    today = commands.validate_command(
        intent_parser.parse_intent("what should I focus on today?"))
    weekly = commands.validate_command(
        intent_parser.parse_intent("what should I focus on this week?"))
    assert today["intent"] == "list" and today["focus_intelligence"] is True
    assert today["due_today"] is True
    assert weekly["intent"] == "weekly_focus"


def test_weekly_focus_uses_current_week_and_overdue_carryover(slack, monkeypatch):
    wednesday = date(2026, 9, 23)
    monkeypatch.setattr(main, "current_date", lambda: wednesday)
    slack.add("Overdue carryover", assignee="UA", priority="P2",
              due=(wednesday - timedelta(days=2)).isoformat())
    slack.add("Friday P1", assignee="UA", priority="P1",
              due=(wednesday + timedelta(days=2)).isoformat())
    slack.add("Sunday deadline", assignee="UA", priority="P3",
              due=(wednesday + timedelta(days=4)).isoformat())
    slack.add("Next Monday", assignee="UA", priority="P1",
              due=(wednesday + timedelta(days=5)).isoformat())
    slack.add("Completed this week", assignee="UA", priority="P1", completed=True,
              due=(wednesday + timedelta(days=1)).isoformat())

    response = ask("what should I focus on this week?", thread="WEEKLY_FOCUS")

    assert response.startswith("*🎯 Weekly Focus*\n\n*Sep 21–Sep 27*\n\n")
    assert "*🔴 Overdue Carryover*" in response and "Overdue carryover" in response
    assert "*📅 Due This Week*" in response
    assert response.index("Friday P1") < response.index("Sunday deadline")
    assert "Next Monday" not in response and "Completed this week" not in response
    assert not slack.writes


def test_risk_analysis_is_factual_and_excludes_normal_future_and_completed(slack, monkeypatch):
    today = main.current_date()
    monkeypatch.setattr(slack_tools, "user_display_name", lambda user_id: "AasthaA" if user_id == "UA" else None)
    slack.add("Overdue risk", assignee="UA", priority="P3", due=(today - timedelta(days=2)).isoformat())
    slack.add("Urgent risk", assignee="UA", priority="P1", due=today.isoformat())
    slack.add("Unassigned risk", priority="P1", due=(today + timedelta(days=10)).isoformat())
    slack.add("Normal future", assignee="UA", priority="P3", due=(today + timedelta(days=10)).isoformat())
    slack.add("Completed risk", completed=True, assignee="UA", priority="P1", due=today.isoformat())
    response = ask("show risky tasks", thread="INTELLIGENCE_RISK")
    assert "Overdue risk" in response and "Overdue by 2 days" in response
    assert "Urgent risk" in response and "Due today" in response
    assert "Unassigned risk" in response and "No assigned owner" in response
    assert "Normal future" not in response and "Completed risk" not in response
    assert not slack.writes


def test_team_workload_counts_priority_overdue_due_soon_and_unassigned(slack, monkeypatch):
    today = main.current_date()
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: {"UA": "AasthaA", "UP": "Praveen"}.get(user_id))
    slack.add("P one", assignee="UP", priority="P1", due=(today - timedelta(days=1)).isoformat())
    slack.add("P two", assignee="UP", priority="P2", due=(today + timedelta(days=1)).isoformat())
    slack.add("P three", assignee="UP", priority="P3", due=(today + timedelta(days=8)).isoformat())
    slack.add("Needs owner", priority="P1", due=(today + timedelta(days=1)).isoformat())
    response = ask("show team workload", thread="INTELLIGENCE_WORKLOAD")
    assert "Praveen" in response and re.search(r"Praveen\s+3\s+1\s+1\s+1", response)
    assert "Unassigned" in response and re.search(r"Unassigned\s+1\s+1", response)
    assert not slack.writes


def test_team_workload_empty_state_is_read_only(slack):
    response = ask("workload summary", thread="INTELLIGENCE_WORKLOAD_EMPTY")
    assert response == "*👥 Team Workload* · No matching pending tasks found."
    assert not slack.writes


def test_action_item_health_uses_current_authorized_snapshot_without_ids_or_writes(slack, monkeypatch):
    today = main.current_date()
    monkeypatch.setattr(slack_tools, "user_display_name", lambda user_id: "Praveen" if user_id == "UP" else None)
    slack.add("Health overdue", assignee="UP", priority="P1", due=(today - timedelta(days=1)).isoformat())
    slack.add("Health complete", completed=True, assignee="UP", priority="P2")
    response = ask("health", thread="INTELLIGENCE_HEALTH")
    assert "Action Items Health" in response
    assert "• Pending: 1" in response and "• Completed: 1" in response
    assert "• Overdue: 1" in response and "• P1: 1" in response
    assert "Highest Attention" in response and "Health overdue" in response
    assert "UP" not in response and "I1" not in response
    # One schema read plus one paginated item read; analysis does not refetch.
    assert not slack.writes and len(slack.list_requests) == 2


def test_similar_task_intelligence_reports_metadata_differences_without_merging(slack, monkeypatch):
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: {"UA": "AasthaA", "UP": "Praveen"}.get(user_id))
    due = (main.current_date() + timedelta(days=3)).isoformat()
    slack.add("Prepare client report", assignee="UA", priority="P3")
    slack.add("Prepare client report", assignee="UP", priority="P2", due=due)
    slack.add("Prepare the client report", assignee="UA", priority="P3")
    before = deepcopy(slack.items)
    response = ask("show tasks similar to Prepare client report", thread="INTELLIGENCE_SIMILAR")
    assert "Similar Tasks" in response and "Exact duplicate title" in response
    assert "Owner: AasthaA" in response and "Owner: Praveen" in response
    assert "Similar title" in response and "No tasks were merged or changed" in response
    assert slack.items == before and not slack.writes


def test_weekly_insights_follow_pending_section_and_are_factual(slack, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "intelligence-weekly.sqlite3"))
    monkeypatch.setattr(slack_tools, "user_display_name", lambda user_id: "Praveen" if user_id == "UP" else None)
    today = main.current_date()
    slack.add("Late P1", assignee="UP", priority="P1", due=(today - timedelta(days=1)).isoformat())
    slack.add("Soon P1", assignee="UP", priority="P1", due=(today + timedelta(days=1)).isoformat())
    slack.add("No owner", priority="P2")
    slack.add("Done", completed=True, assignee="UP")
    response = ask("weekly summary", thread="INTELLIGENCE_WEEKLY")
    assert response.index("*Pending · 3*") < response.index("*Insights*")
    assert "1 task is overdue" in response
    assert "1 task is due within 48 hours" in response
    assert "1 pending task is unassigned" in response
    assert "Praveen has 2 pending P1 tasks" in response
    assert "*Pending · 3*" in response and "Overdue:* 1" in response
    assert not slack.writes


def test_dependency_intelligence_parses_named_queries_and_stays_truthful(slack):
    slack.add("Client report", assignee="UA")
    response = ask("what is blocking the client report?", thread="INTELLIGENCE_DEPENDENCY")
    assert "no explicit dependency or blocker field" in response
    parsed = intent_parser.parse_intent("show dependencies for the API documentation task")
    assert parsed["intent"] == "dependencies"
    assert parsed["task_name"] == "API documentation"


@pytest.mark.parametrize("raw,expected", [
    ("P1", "P1"),
    ({"label": "P2"}, "P2"),
    ([{"value": "priority_three"}], "P3"),
    (None, None),
])
def test_priority_normalization_supports_slack_select_shapes(raw, expected):
    schema = {"schema": [{
        "id": "Col0C2BUJ3084", "key": "Col0C23G8MR9B", "name": "Priority",
        "type": "select", "options": {"choices": [
            {"id": "priority_one", "label": "P1"},
            {"id": "priority_two", "label": "P2"},
            {"id": "priority_three", "label": "P3"},
        ]},
    }]}
    assert slack_tools.normalize_priority(raw, schema) == expected


def test_structured_slack_priority_is_shared_by_all_intelligence_calculations(slack):
    today = main.current_date()
    schema = deepcopy(SCHEMA)
    schema["schema"][-1] = {
        "id": "Col0C2BUJ3084", "key": "Col0C23G8MR9B", "name": "Priority",
        "type": "select", "options": {"choices": [
            {"id": "priority_one", "label": "P1"},
            {"id": "priority_two", "label": "P2"},
            {"id": "priority_three", "label": "P3"},
        ]},
    }
    item = {"id": "Rec0C4RAZNGC8", "fields": [
        {"column_id": "name", "text": "Structured priority"},
        {"column_id": "done", "checkbox": False},
        {"column_id": "owner", "user": ["UP"]},
        {"column_id": "due", "date": [today.isoformat()]},
        {"column_id": "Col0C2BUJ3084", "select": [{"value": "priority_one"}]},
    ]}
    readable = main._readable_analysis_schema(
        schema, main.context("UA", "C", "PRIORITY_SCHEMA", None, "W"))
    snapshot = project_intelligence.normalize_task_snapshot([item], readable)
    assert snapshot[0].priority == "P1"
    assert project_intelligence.generate_task_health(snapshot, readable, today).p1 == 1
    assert project_intelligence.calculate_daily_focus(snapshot, readable, today)[0].priority == "P1"
    workload = project_intelligence.calculate_workload(
        snapshot, readable, lambda user_id: "Praveen", today, ["UP"])
    assert workload.rows["UP"]["p1"] == 1


def test_focus_sections_and_footer_count_are_consistent(slack):
    today = main.current_date()
    slack.add("Late one", assignee="UA", priority="P1", due=(today - timedelta(days=1)).isoformat())
    slack.add("Today one", assignee="UA", priority="P2", due=today.isoformat())
    slack.add("Tomorrow one", assignee="UA", priority="P1", due=(today + timedelta(days=1)).isoformat())
    slack.add("Upcoming one", assignee="UA", priority="P2", due=(today + timedelta(days=4)).isoformat())
    response = ask("what should I focus on today?", thread="FOCUS_SECTIONS")
    assert "*🔴 Immediate Attention*" in response and "*📅 Upcoming*" in response
    assert "*Summary*\n\n2 tasks require immediate attention." in response
    assert response.index("Late one") < response.index("Today one") < response.index("Tomorrow one")


def test_focus_today_has_explicit_slack_line_breaks_between_sections_and_tasks(
        slack, monkeypatch):
    today = main.current_date()
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: "AasthaA" if user_id == "UA" else None)
    slack.add("First overdue", assignee="UA", priority="P1",
              due=(today - timedelta(days=4)).isoformat())
    slack.add("Second overdue", assignee="UA", priority="P2",
              due=(today - timedelta(days=2)).isoformat())
    slack.add("Upcoming focus", assignee="UA", priority="P1",
              due=(today + timedelta(days=2)).isoformat())
    rendered = ask("what should I focus on today?", thread="FOCUS_FORMAT")

    assert rendered.startswith("*🎯 Focus Today*\n\n")
    assert "No action items are due today.\n\n*Pending work:* 3 tasks" in rendered
    assert "*🔴 Immediate Attention*\n\n```\nTask" in rendered
    assert "First overdue" in rendered and "🔴 4d overdue" in rendered
    assert "Second overdue" in rendered
    assert "*📅 Upcoming*\n\n```\nTask" in rendered and "Upcoming focus" in rendered
    assert "\n\n*Summary*\n\n2 tasks require immediate attention." in rendered
    assert "today.3" not in rendered and "Attention*1." not in rendered
    assert "days2." not in rendered and "**" not in rendered

    posted = {}
    client = SimpleNamespace(
        chat_postMessage=lambda **kwargs: posted.update(kwargs) or {"ok": True})
    monkeypatch.setattr(main, "app", SimpleNamespace(client=client))
    main.post("C", rendered, thread_ts="FOCUS_FORMAT")
    assert posted["text"] == rendered
    assert posted["thread_ts"] == "FOCUS_FORMAT"


def test_weekly_focus_final_payload_preserves_section_and_item_boundaries(
        slack, monkeypatch):
    today = main.current_date()
    monkeypatch.setattr(
        slack_tools, "user_display_name",
        lambda user_id: "AasthaA" if user_id == "UA" else None)
    slack.add("Weekly overdue one", assignee="UA", priority="P2",
              due=(today - timedelta(days=2)).isoformat())
    slack.add("Weekly overdue two", assignee="UA", priority="P2",
              due=(today - timedelta(days=1)).isoformat())
    rendered = ask("what should I focus on this week?", thread="WEEKLY_FORMAT")

    assert rendered.startswith("*🎯 Weekly Focus*\n\n*")
    assert "*🔴 Overdue Carryover*\n\n```\nTask" in rendered
    assert "Weekly overdue one" in rendered and "Weekly overdue two" in rendered
    assert "\n\n*Summary*\n\n2 pending tasks require attention this week." in rendered
    assert "🎯Sep" not in rendered and "Carryover*•" not in rendered
    assert "week.Showing" not in rendered and "**" not in rendered

    posted = {}
    client = SimpleNamespace(
        chat_postMessage=lambda **kwargs: posted.update(kwargs) or {"ok": True})
    monkeypatch.setattr(main, "app", SimpleNamespace(client=client))
    main.post("C", rendered, thread_ts="WEEKLY_FORMAT")
    assert posted["text"] == rendered
    assert posted["thread_ts"] == "WEEKLY_FORMAT"


def test_intelligence_formatting_uses_slack_mrkdwn_and_hides_internal_ids(slack):
    today = main.current_date()
    slack.add("Formatting task", assignee="UP", priority="P1", due=today.isoformat())
    responses = [
        ask("health", thread="FORMAT_HEALTH"),
        ask("show team workload", thread="FORMAT_WORKLOAD"),
        ask("show risky tasks", thread="FORMAT_RISK"),
    ]
    for response in responses:
        assert "**" not in response
        assert "UP" not in response and "I1" not in response
        assert "{" not in response and "}" not in response


def test_complex_goal_creates_approved_plan_without_automatic_action(slack, monkeypatch):
    today = main.current_date()
    slack.add("API release verification", assignee="UP", priority="P1",
              due=(today - timedelta(days=1)).isoformat())
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: sent.append(args))

    response = ask("Prepare everything needed for Friday's API release.", thread="ORCH_CREATE")
    assert response.startswith("*Execution Plan*")
    assert "API release verification" in response
    assert "*Approval Required*" in response and "ready for approval" in response and "approve plan" in response
    assert "No actions have been taken" in response
    assert not sent and not slack.writes


def test_orchestrator_approval_executes_once_and_verifies(slack, monkeypatch):
    today = main.current_date()
    slack.add("Deployment readiness", assignee="UP", priority="P1",
              due=(today - timedelta(days=1)).isoformat())
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: sent.append(args))
    preview = ask("Get the team ready for tomorrow's deployment.", thread="ORCH_APPROVE")
    plan_id = re.search(r"approve plan ([a-f0-9]{8})", preview).group(1)

    executed = ask(f"approve plan {plan_id}", thread="ORCH_APPROVE")
    duplicate = ask(f"approve plan {plan_id}", thread="ORCH_APPROVE")
    assert "Plan Executed" in executed and "Reminder sent" in executed
    assert "Plan Already Processed" in duplicate
    assert len(sent) == 1 and sent[0][0] == "UP"


def test_orchestrator_rejects_stale_plan_before_any_action(slack, monkeypatch):
    today = main.current_date()
    item = slack.add("Overdue cleanup", assignee="UP", priority="P1",
                     due=(today - timedelta(days=1)).isoformat())
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: sent.append(args))
    preview = ask("Help me clean up our overdue work.", thread="ORCH_STALE")
    plan_id = re.search(r"approve plan ([a-f0-9]{8})", preview).group(1)
    next(field for field in item["fields"] if field["column_id"] == "priority")["select"] = ["priority_2"]

    response = ask(f"approve plan {plan_id}", thread="ORCH_STALE")
    assert "Plan Expired" in response and "No actions were executed" in response
    assert not sent


def test_orchestrator_failure_is_reported_without_crashing_other_logic(slack, monkeypatch):
    today = main.current_date()
    slack.add("Risk follow-up", assignee="UP", priority="P1",
              due=(today - timedelta(days=1)).isoformat())
    monkeypatch.setattr(main, "_send_deadline_reminder",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("send failed")))
    preview = ask("Prepare the actions needed to reduce the current risks.", thread="ORCH_FAIL")
    plan_id = re.search(r"approve plan ([a-f0-9]{8})", preview).group(1)
    response = ask(f"approve plan {plan_id}", thread="ORCH_FAIL")
    assert "Plan Not Completed" in response and "reminder delivery failed" in response


def test_orchestrator_contextual_followups_are_safe_and_ambiguous_plans_clarify(slack):
    today = main.current_date()
    slack.add("Single overdue task", assignee="UA", priority="P1",
              due=(today - timedelta(days=1)).isoformat())
    first = ask("Help me clean up our overdue work.", thread="ORCH_CONTEXT")
    explanation = ask("Why did you include that reminder?", thread="ORCH_CONTEXT")
    removed = ask("Remove that step", thread="ORCH_CONTEXT")
    assert "Reason:" in explanation and "No action has been taken" in explanation
    assert "Removed the step" in removed
    assert "Execution Plan" in first

    ask("Help me clean up our overdue work.", thread="ORCH_MULTI")
    ask("Review our current situation and prepare the next steps.", thread="ORCH_MULTI")
    ambiguous = ask("now approve it", thread="ORCH_MULTI")
    assert "Multiple plans are active" in ambiguous


def test_orchestrator_rbac_marks_unauthorized_step_and_read_only_plan_needs_no_approval(
        slack, monkeypatch):
    today = main.current_date()
    slack.add("Viewer overdue", assignee="UV", priority="P1",
              due=(today - timedelta(days=1)).isoformat())
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: sent.append(args))
    restricted = ask("Help me clean up our overdue work.", user="UV", thread="ORCH_RBAC")
    assert "not authorized for execution" in restricted
    assert "No executable actions require approval" in restricted
    assert not sent

    slack.items.clear()
    read_only = ask("Review our current situation and prepare the next steps.",
                    thread="ORCH_READ_ONLY")
    assert "No executable actions require approval" in read_only
    assert "No changes were made" in read_only


def test_orchestrator_expired_plan_is_rejected(slack):
    today = main.current_date()
    slack.add("Expired plan task", assignee="UA", priority="P1",
              due=(today - timedelta(days=1)).isoformat())
    preview = ask("Help me clean up our overdue work.", thread="ORCH_EXPIRED")
    plan_id = re.search(r"approve plan ([a-f0-9]{8})", preview).group(1)
    ctx = main.context("UA", "C", "ORCH_EXPIRED", None, "W")
    state = main._state(ctx)
    state["orchestrator_plans"][plan_id]["expires_at"] = 0
    main._save_state(ctx, state)
    response = ask(f"approve plan {plan_id}", thread="ORCH_EXPIRED")
    assert "expired" in response.casefold() and "no actions were taken" in response.casefold()


@pytest.mark.parametrize("goal,context_type", [
    ("prepare everything needed for the current overdue work", "overdue_work"),
    ("prepare actions for overdue tasks", "overdue_work"),
    ("what is overdue", "overdue_work"),
    ("review overdue items", "overdue_work"),
    ("prepare tasks due today", "due_today"),
    ("prepare what is due today", "due_today"),
    ("prepare work due soon", "deadline_risk"),
    ("prepare tasks within 24 hours", "due_24h"),
    ("prepare tasks within 48 hours", "due_48h"),
    ("prepare P1 tasks", "priority_p1"),
    ("prepare unassigned work", "unassigned_tasks"),
    ("prepare actions for current risks", "current_risks"),
    ("review team workload", "team_workload"),
])
def test_orchestrator_goal_context_resolution_is_deterministic(goal, context_type):
    assert agent_orchestrator.resolve_goal_context(goal) == context_type


def test_overdue_orchestration_uses_concrete_authorized_tasks_and_autopilot(slack, monkeypatch):
    today = main.current_date()
    slack.add("Late P1 report", assignee="UP", priority="P1",
              due=(today - timedelta(days=2)).isoformat())
    slack.add("Late unassigned checklist", priority="P2",
              due=(today - timedelta(days=1)).isoformat())
    slack.add("Future unrelated task", assignee="UA", priority="P1",
              due=(today + timedelta(days=5)).isoformat())
    sent = []
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: sent.append(args))

    response = ask("prepare everything needed for the current overdue work",
                   thread="ORCH_OVERDUE_CONCRETE")
    assert "*Overdue Work*" in response
    assert "2 overdue pending tasks" in response
    assert "Late P1 report" in response and "Praveen" in response and "P1" in response
    assert "Late unassigned checklist" in response and "Unassigned" in response
    assert "Future unrelated task" not in response
    assert "Prepare reminder" in response and "Review ownership" in response
    assert "Review the current authorized task state" not in response
    assert "1 action is ready for approval" in response
    assert not sent

    plan_id = re.search(r"approve plan ([a-f0-9]{8})", response).group(1)
    executed = ask(f"approve plan {plan_id}", thread="ORCH_OVERDUE_CONCRETE")
    assert "Plan Executed" in executed and len(sent) == 1
    # The message is the existing Autopilot wording, not a second reminder prompt.
    assert "still pending" in sent[0][1] and "due 2 days ago" in sent[0][1]


def test_overdue_orchestration_has_truthful_empty_state(slack):
    slack.add("Future task", assignee="UA", priority="P2",
              due=(main.current_date() + timedelta(days=3)).isoformat())
    response = ask("prepare everything needed for the current overdue work",
                   thread="ORCH_OVERDUE_EMPTY")
    assert "No overdue pending tasks found" in response
    assert "No action is currently required" in response
    assert "No executable actions require approval" in response
    assert "1 relevant pending task" not in response


def test_overdue_orchestration_respects_member_rbac(slack):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    slack.add("Member overdue", assignee="UM", priority="P1", due=yesterday)
    slack.add("Admin overdue", assignee="UA", priority="P1", due=yesterday)
    response = ask("prepare everything needed for the current overdue work",
                   user="UM", thread="ORCH_OVERDUE_RBAC")
    assert "Member overdue" in response and "Admin overdue" not in response
    assert "UA" not in response and "UM" not in response


def test_orchestrator_numbered_followups_and_robot_free_ux(slack):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    slack.add("Numbered overdue", assignee="UA", priority="P1", due=yesterday)
    preview = ask("prepare actions for the overdue tasks", thread="ORCH_NUMBERED")
    explained = ask("why did you include step 1?", thread="ORCH_NUMBERED")
    removed = ask("remove step 1", thread="ORCH_NUMBERED")
    assert "Reason:" in explained and "Removed the step" in removed
    for rendered in (preview, explained, removed):
        assert chr(0x1F916) not in rendered
        assert (":" + "robot" + "_" + "face:") not in rendered
        assert "**" not in rendered
    assert preview.startswith("*Execution Plan*")


def test_orchestrator_deduplicates_aggregate_workload_into_team_insight(slack):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    slack.add("Owner A urgent one", assignee="UA", priority="P1", due=yesterday)
    slack.add("Owner A urgent two", assignee="UA", priority="P1", due=yesterday)
    slack.add("Owner B urgent one", assignee="UP", priority="P1", due=yesterday)
    slack.add("Owner B urgent two", assignee="UP", priority="P1", due=yesterday)

    response = ask("prepare everything needed for the current overdue work",
                   thread="ORCH_AGGREGATE_DEDUP")
    prepared = response.split("*Prepared Actions*", 1)[1].split("*Team Insight*", 1)[0]
    insight = response.split("*Team Insight*", 1)[1].split("*Risk*", 1)[0]

    assert prepared.count("Prepare reminder") == 4
    assert "Review workload" not in prepared
    assert insight.count("Multiple urgent action items are creating workload pressure") == 1
    assert "4 actions are ready for approval" in response
    assert "**" not in response and chr(0x1F916) not in response


def test_overdue_orchestration_preserves_three_concrete_tasks(slack):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    for name in ("Follow-up Test", "Follow-up Test Task", "Reminder Docs Task"):
        slack.add(name, assignee="UA", priority="P1", due=yesterday)
    response = ask("prepare everything needed for the current overdue work",
                   thread="ORCH_THREE_TASKS")
    assert "3 overdue pending tasks" in response
    assert all(name in response for name in (
        "Follow-up Test", "Follow-up Test Task", "Reminder Docs Task"))
    assert response.count("Prepare reminder") == 3
    assert "3 actions are ready for approval" in response


def test_orchestrator_plan_presentation_is_compact_slack_mrkdwn(slack):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    slack.add("Compact task", assignee="UP", priority="P1", due=yesterday)
    response = ask("prepare everything needed for the current overdue work",
                   thread="ORCH_PRESENTATION")
    assert response.startswith("*Execution Plan*\n\n*Goal*")
    assert "\n\n*Current State*\n" in response
    assert "\n\n*Overdue Work*\n• *Compact task* · P1 · Praveen · due " in response
    assert "\n\n*Prepared Actions*\n1. Prepare reminder for *Compact task*" in response
    assert "\n\n*Approval Required*\n1 action is ready for approval." in response
    assert response.endswith("*No actions have been taken.*")
    assert "*Team Insight*" not in response
    assert "**" not in response and chr(0x1F916) not in response


def test_actual_orchestrator_chat_post_message_payload_is_native_slack_mrkdwn(
        slack, monkeypatch, caplog):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    slack.add("Final payload task", assignee="UP", priority="P1", due=yesterday)
    posted = []

    class Client:
        @staticmethod
        def chat_postMessage(**kwargs):
            posted.append(kwargs)
            return {"ok": True, "ts": "900.1"}

    rendered = []
    render = main._render_orchestrator_slack_message

    def capture_rendered_plan(plan):
        result = render(plan)
        rendered.append(result)
        return result

    monkeypatch.setattr(main, "_render_orchestrator_slack_message", capture_rendered_plan)
    monkeypatch.setattr(main, "app", SimpleNamespace(client=Client()))
    caplog.set_level(logging.WARNING, logger="slack_list")
    main._deliver(
        "event:orchestrator-final-payload",
        "prepare everything needed for the current overdue work",
        "UA", "C", thread_ts="THREAD", msg_ts="REQUEST", team_id="W")

    assert len(posted) == 1
    payload = posted[0]["text"]
    assert rendered == [payload]
    assert "blocks" not in posted[0]
    assert "*Execution Plan*" in payload
    assert "*Goal*" in payload and "*Prepared Actions*" in payload
    assert "Final payload task" in payload
    assert "Praveen" in payload
    assert main.slack_presentation.compact_date(yesterday, main.current_date()) in payload
    plan_id = re.search(r"approve plan ([a-f0-9]{8})", payload).group(1)
    assert f"`approve plan {plan_id}`" in payload
    assert "**" not in payload
    assert ("**" + "Execution Plan" + "**") not in payload
    assert ("slack-" + "edge.com") not in payload
    assert (":" + "red" + "_circle:") not in payload
    assert (":" + "robot" + "_face:") not in payload
    assert chr(0x1F916) not in payload
    assert posted[0]["channel"] == "C" and posted[0]["thread_ts"] == "THREAD"
    assert not slack.writes
    boundary_log = next(
        record.getMessage() for record in caplog.records
        if "ORCHESTRATOR_FINAL_SLACK_TEXT" in record.getMessage()
    )
    assert "main.post->app.client.chat_postMessage" in boundary_log
    assert "main._render_orchestrator_slack_message" in boundary_log
    assert repr(payload) in boundary_log
    payload_log = next(
        record.getMessage() for record in caplog.records
        if "ORCHESTRATOR_CHAT_POST_PAYLOAD" in record.getMessage()
    )
    assert "main.post->app.client.chat_postMessage" in payload_log
    assert f"text={payload!r}" in payload_log
    assert "blocks=None" in payload_log


def test_orchestrator_seven_rows_keep_insights_separate_and_count_only_actions(slack):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    for index in range(7):
        slack.add(f"Dynamic overdue {index + 1}", assignee="UA", priority="P1",
                  due=yesterday)

    response = ask("prepare everything needed for the current overdue work",
                   thread="ORCH_SEVEN_ACTIONS")
    prepared = response.split("*Prepared Actions*", 1)[1].split("*Team Insight*", 1)[0]
    insight = response.split("*Team Insight*", 1)[1].split("*Risk*", 1)[0]

    assert "7 overdue pending tasks" in response
    assert len(re.findall(r"^\d+\. (?:Prepare reminder for|Review status for) "
                          r"\*Dynamic overdue \d+\*$", prepared, re.M)) == 7
    assert prepared.count("Prepare reminder for") == 5
    assert "Review workload" not in prepared
    assert "Multiple urgent action items are creating workload pressure." in insight
    assert "5 actions are ready for approval." in response
    assert re.search(r"`approve plan [a-f0-9]{8}`", response)
    assert not slack.writes


def test_orchestrator_execution_and_failure_presentations_are_concise(slack, monkeypatch):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    slack.add("Delivery success", assignee="UP", priority="P1", due=yesterday)
    monkeypatch.setattr(main, "_send_deadline_reminder", lambda *args: {"ok": True, "ts": "1"})
    preview = ask("prepare actions for the overdue tasks", thread="ORCH_RESULT_OK")
    plan_id = re.search(r"approve plan ([a-f0-9]{8})", preview).group(1)
    success = ask(f"approve plan {plan_id}", thread="ORCH_RESULT_OK")
    assert success.startswith("*Plan Executed*\n\n*Completed*")
    assert "Reminder sent to Praveen for *Delivery success*" in success
    assert "*Failed*" not in success and "🟢 All approved actions verified" in success
    assert "**" not in success

    slack.items.clear()
    slack.add("Delivery failure", assignee="UP", priority="P1", due=yesterday)
    monkeypatch.setattr(main, "_send_deadline_reminder",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("private details")))
    preview = ask("prepare actions for the overdue tasks", thread="ORCH_RESULT_FAIL")
    plan_id = re.search(r"approve plan ([a-f0-9]{8})", preview).group(1)
    failure = ask(f"approve plan {plan_id}", thread="ORCH_RESULT_FAIL")
    assert failure.startswith("*Plan Not Completed*")
    assert "*Failed*\n• Reminder for *Delivery failure* · reminder delivery failed" in failure
    assert "private details" not in failure and "Traceback" not in failure


def test_orchestrator_long_plan_is_bounded_but_explicit(slack):
    yesterday = (main.current_date() - timedelta(days=1)).isoformat()
    for index in range(12):
        slack.add(f"Long overdue {index + 1}", assignee="UA", priority="P1", due=yesterday)
    response = ask("prepare everything needed for the current overdue work",
                   thread="ORCH_LONG_PRESENTATION")
    overdue_section = response.split("*Overdue Work*", 1)[1].split("*Prepared Actions*", 1)[0]
    prepared_section = response.split("*Prepared Actions*", 1)[1].split("*Team Insight*", 1)[0]
    assert overdue_section.count("• *Long overdue") == 10
    assert len(re.findall(r"^\d+\. ", prepared_section, re.M)) == 10
    assert "2 additional prepared actions are stored in this plan" in prepared_section


@pytest.mark.parametrize("phrase,expected", [
    ("list my tasks", "list"),
    ("show me our workload", "workload"),
    ("command center", "command_center"),
    ("visualize our workload", "visual_analytics"),
])
def test_simple_requests_bypass_orchestrator(phrase, expected):
    assert intent_parser.parse_intent(phrase)["intent"] == expected


@pytest.mark.parametrize("phrase,operation", [
    ("what happens if I assign the unassigned P1 task to Praveen?", "assign_task"),
    ("simulate assigning the unassigned P1 to AasthaA", "assign_task"),
    ("what happens if we move the API documentation deadline to Friday?", "change_due_date"),
    ("what happens if we make this P1?", "change_priority"),
    ("what happens if we do nothing?", "leave_unchanged"),
    ("simulate the safest way to reduce our current deadline risk", "workload_redistribution"),
])
def test_simulation_requests_route_deterministically_without_llm(phrase, operation, monkeypatch):
    monkeypatch.setattr(intent_parser, "_configured_ollama_client",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("simulation should be deterministic")))
    parsed = intent_parser.parse_intent(phrase)
    assert parsed["intent"] == "simulation"
    assert parsed["scenario"]["operation"] == operation


def test_simulation_is_read_only_persisted_and_explains_tradeoffs(slack, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "simulation.sqlite3"))
    slack.add("Unassigned P1 release", priority="P1", due=main.current_date().isoformat())
    slack.add("Existing Praveen work", assignee="UP", priority="P1",
              due=(main.current_date() + timedelta(days=1)).isoformat())

    response = ask("what happens if I assign the unassigned P1 task to Praveen?",
                   thread="SIM_READ_ONLY")

    assert response.startswith("*🔎 Scenario Simulation*")
    assert "*Current State*" in response and "*Projected State*" in response
    assert "Removes 1 unassigned task" in response
    assert "pending workload increases" in response
    assert "*Risk Impact*" in response and "*Assumptions*" in response
    assert "No changes were made" in response
    assert not slack.writes
    scenario_id = re.search(r"Scenario · `([a-f0-9]{10})`", response).group(1)
    shown = ask(f"show simulation {scenario_id}", thread="SIM_READ_ONLY")
    assert "*Decision ·" in shown and "expected outcome remains frozen" in shown
    history = ask("show my recent simulations", thread="SIM_READ_ONLY")
    assert "*Decision History*" in history and "assign the unassigned P1" in history


def test_simulation_prepare_requires_separate_approval_and_detects_stale_state(
        slack, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "simulation-stale.sqlite3"))
    item = slack.add("Unassigned P1 release", priority="P1", due=main.current_date().isoformat())
    ask("simulate assigning the unassigned P1 to Praveen", thread="SIM_STALE")
    next(field for field in item["fields"] if field["column_id"] == "priority")["select"] = ["priority_2"]
    response = ask("prepare this scenario", thread="SIM_STALE")
    assert "Scenario is stale" in response
    assert not slack.writes


def test_simulation_prepare_bridges_to_existing_proposal_without_writing(
        slack, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "simulation-prepare.sqlite3"))
    slack.add("Unassigned P1 release", priority="P1", due=main.current_date().isoformat())
    simulation = ask("simulate assigning the unassigned P1 to Praveen", thread="SIM_PREPARE")
    prepared = ask("prepare this scenario", thread="SIM_PREPARE")
    assert "Scenario Simulation" in simulation
    assert "*Scenario Prepared — Approval Required*" in prepared
    assert "apply this proposal" in prepared
    assert not slack.writes


def test_simulation_approval_reuses_executor_and_records_actual_outcome(
        slack, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "simulation-execute.sqlite3"))
    slack.add("Unassigned P1 release", priority="P1", due=main.current_date().isoformat())
    simulation = ask("simulate assigning the unassigned P1 to Praveen", thread="SIM_EXECUTE")
    decision_id = re.search(r"Decision · `([a-f0-9]{12})`", simulation).group(1)
    ask("prepare this scenario", thread="SIM_EXECUTE")
    applied = ask("apply this proposal", thread="SIM_EXECUTE")
    repeated = ask("apply this proposal", thread="SIM_EXECUTE")
    verification = ask(
        f"compare expected vs actual for decision {decision_id}", thread="SIM_EXECUTE")

    assert "Proposal applied and verified" in applied
    assert len(slack.writes) == 1
    assert "There is no current proposal" in repeated and len(slack.writes) == 1
    assert "*Result:* Verified" in verification


def test_simulation_comparison_uses_same_authorized_task_and_never_mutates(
        slack, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "simulation-compare.sqlite3"))
    task = slack.add("Comparison task", priority="P1", due=main.current_date().isoformat())
    main.store_view("unused", [task], SCHEMA,
                    main.context("UA", "C", "SIM_COMPARE", None, "W"))
    response = ask("compare assigning this to Praveen vs AasthaA", thread="SIM_COMPARE")
    assert response.startswith("*🔎 Scenario Comparison*")
    assert "Praveen" in response and "Aastha" in response
    assert "No option was automatically selected" in response
    assert not slack.writes
