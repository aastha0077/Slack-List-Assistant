import base64
import json
import logging
import os
import re
import sqlite3
import threading
import time
import tempfile
from dataclasses import replace
from datetime import date, timedelta, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import config
import slack_tools
import mutations
import delivery
import progress_engine
import slack_presentation
import project_intelligence
import predictive_intelligence
import workflow_safety
import audit_log
import content_ingestion
import transcription
import action_item_extraction
import source_trace
import visual_analytics
import deadline_reminders
import action_item_sentinel
import smart_task_autopilot
import command_center
import agent_orchestrator
import task_simulation
import decision_ledger
from safe_diagnostics import redact
from commands import validate_command
from intent_parser import parse_intent
from references import (parse_reference, reference_from, select_ids, TargetType, ResolvedTargetSet,
                        requested_cardinality, resolved_target_type, enforce_cardinality)
from copy import deepcopy

def current_date():
    return datetime.now(ZoneInfo("Asia/Kathmandu")).date()

def normalize_task_name(name):
    return re.sub(r"\s+", " ", name).casefold().strip()

load_dotenv()
BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "").strip()
APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "").strip()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("slack_list")
app = None  # Constructed only by create_app(), never during imports.

DB_PATH = os.getenv("STATE_DB", "slack_list_state.sqlite3")
_db_lock = threading.Lock()
# Diagnostic mirror only; SQLite is authoritative, including after restart.
_pending = {}

OUT_OF_SCOPE = "Sorry, I can only help with Slack List and action-item related requests."

# How long (seconds) to keep thread context: 3 hours
_CONTEXT_TTL = 10800
_CLARIFICATION_TTL = 900


def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_context (
            ctx_key TEXT PRIMARY KEY,
            data    TEXT NOT NULL,
            created REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Persistent context helpers
# ---------------------------------------------------------------------------

def _ctx_write(key: str, entry: dict):
    """Stage during delivery; only published responses become conversation state."""
    if delivery.is_active():
        delivery.stage_context(key, entry)
        return
    try:
        with _db_lock:
            conn = _db()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO thread_context(ctx_key, data, created) VALUES (?, ?, ?)",
                    (key, json.dumps(entry, default=str), entry.get("created", time.time()))
                )
                conn.commit()
                _pending[key] = deepcopy(entry)
            finally:
                conn.close()
    except Exception as exc:
        raise RuntimeError("Could not persist conversation context") from exc


def _ctx_read(key: str) -> dict:
    """Read a request's staged state or committed SQLite state with one TTL."""
    staged = delivery.staged_context().get(key) if delivery.is_active() else None
    if staged and time.time() - staged.get("created", 0) <= _CONTEXT_TTL:
        return deepcopy(staged)
    try:
        with _db_lock:
            conn = _db()
            try:
                row = conn.execute(
                    "SELECT data FROM thread_context WHERE ctx_key=? AND created>=?",
                    (key, time.time() - _CONTEXT_TTL)
                ).fetchone()
            finally:
                conn.close()
        if row:
            entry = json.loads(row[0])
            _pending[key] = entry  # warm cache
            return entry
    except Exception as exc:
        raise RuntimeError("Could not read conversation context") from exc
    return {}


def _publish_context():
    entries = delivery.staged_context()
    if not entries:
        return
    with _db_lock:
        conn = _db()
        try:
            conn.executemany("INSERT OR REPLACE INTO thread_context VALUES (?, ?, ?)",
                             [(key, json.dumps(entry), entry["created"]) for key, entry in entries.items()])
            conn.commit()
            _pending.update(deepcopy(entries))
        finally:
            conn.close()


def _ctx_delete_key(key: str):
    """Remove one key from cache and SQLite."""
    _pending.pop(key, None)
    try:
        with _db_lock:
            conn = _db()
            try:
                conn.execute("DELETE FROM thread_context WHERE ctx_key=?", (key,))
                conn.commit()
            finally:
                conn.close()
    except Exception as exc:
        logger.warning("ctx_delete_key failed for key %r: %s", key, exc)


def _ctx_cleanup():
    """Evict expired entries from cache and SQLite."""
    cutoff = time.time() - _CONTEXT_TTL
    for k in [k for k, v in list(_pending.items()) if v.get("created", 0) < cutoff]:
        _pending.pop(k, None)
    try:
        with _db_lock:
            conn = _db()
            try:
                conn.execute("DELETE FROM thread_context WHERE created<?", (cutoff,))
                conn.commit()
            finally:
                conn.close()
    except Exception as exc:
        logger.warning("ctx_cleanup failed: %s", exc)




def context(user_id, channel_id, thread_ts=None, msg_ts=None, team_id=None):
    return config.build_context(user_id=user_id, channel_id=channel_id, thread_ts=thread_ts,
                                msg_ts=msg_ts, team_id=team_id)


def slack_mrkdwn(text):
    """Normalize common Markdown bold syntax at the single Slack output boundary."""
    value = re.sub(r"\\\*\\\*(.+?)\\\*\\\*", r"*\1*", str(text or ""))
    return slack_presentation.validate_slack_response(
        value, resolve_user=user_name)[0]


def _validated_slack_text(text, *, call_site):
    """Apply the presentation-only quality gate and log safe issue codes."""
    validated, issues = slack_presentation.validate_slack_response(
        text, resolve_user=user_name)
    if issues:
        logger.warning(
            "response_quality_issues call_site=%s issue_codes=%s response_chars=%d",
            call_site, ",".join(issues), len(validated))
    logger.info(
        "response_layout_validated call_site=%s line_breaks=%d section_breaks=%d "
        "response_chars=%d",
        call_site, validated.count("\n"), validated.count("\n\n"), len(validated))
    return validated


def post(channel, text, thread_ts=None, metadata=None):
    raw_text = str(text or "")
    renderer = (
        "main._render_orchestrator_slack_message"
        if raw_text.startswith("*Execution Plan*")
        else "main.post"
    )
    # The orchestrator renderer already emits native Slack mrkdwn. Preserve
    # that verified payload byte-for-byte at the final Slack boundary.
    outgoing_text = _validated_slack_text(
        raw_text, call_site="main.post->app.client.chat_postMessage")
    logger.debug(
        "slack_outgoing renderer=%s response_length=%d text_preview=%r",
        renderer, len(outgoing_text), outgoing_text[:500],
    )
    if renderer == "main._render_orchestrator_slack_message":
        logger.warning(
            "ORCHESTRATOR_FINAL_SLACK_TEXT call_site=main.post->app.client.chat_postMessage "
            "renderer=%s response_length=%d repr=%r",
            renderer, len(outgoing_text), outgoing_text,
        )
    kwargs = {"channel": channel, "text": outgoing_text}
    if metadata:
        kwargs["metadata"] = metadata
    if thread_ts: kwargs["thread_ts"] = thread_ts
    if renderer == "main._render_orchestrator_slack_message":
        logger.warning(
            "ORCHESTRATOR_CHAT_POST_PAYLOAD "
            "call_site=main.post->app.client.chat_postMessage text=%r blocks=%r",
            kwargs["text"], kwargs.get("blocks"),
        )
    return app.client.chat_postMessage(**kwargs)


def user_name(uid):
    name = slack_tools.user_display_name(uid)
    if name and str(name).strip().casefold() != str(uid or "").strip().casefold():
        return str(name).strip()
    return "Workspace member"


def item_assignees(item, schema):
    """Return user-facing assignee names without exposing Slack user IDs."""
    return ", ".join(user_name(uid) for uid in slack_tools.extract_assignee_ids(item, schema))


def resolve_follow_up_timezone(owner_id, settings):
    """Resolve one reminder timezone without consulting the host OS timezone."""
    if settings.workspace_timezone:
        timezone, source = settings.workspace_timezone, "team_timezone"
    else:
        profile_timezone = slack_tools.user_timezone(owner_id) if owner_id else None
        if profile_timezone:
            timezone, source = profile_timezone, "slack_profile"
        elif settings.timezone and settings.application_timezone_configured:
            timezone, source = settings.timezone, "application_fallback"
        else:
            timezone, source = settings.timezone or "Asia/Kathmandu", "default_fallback"
    logger.info("follow_up_timezone_resolved user_id=%s timezone=%s source=%s",
                owner_id or "unassigned", timezone, source)
    return timezone, source


def _load_deadline_reminder_tasks(settings=None, sentinel_engine=None):
    """Build owner-scoped reminder rows using existing Slack List readers and RBAC."""
    settings = settings or deadline_reminders.ReminderSettings.from_env()
    list_id = config.DEFAULT_LIST_ID
    if not list_id:
        raise RuntimeError("No Slack List is configured for deadline reminders")
    schema = slack_tools.get_list_schema(list_id)
    items = slack_tools.list_action_items(list_id=list_id)
    if sentinel_engine and sentinel_engine.settings.enabled:
        try:
            snapshot = project_intelligence.normalize_task_snapshot(items, schema)
            sentinel_engine.evaluate(
                list_id, snapshot, datetime.now(ZoneInfo(settings.timezone)),
                persist_snapshot=True)
            logger.info("sentinel_snapshot_fetched list_id=%s task_count=%d", list_id, len(snapshot))
        except Exception:
            logger.exception("sentinel_evaluation_failed list_id=%s", list_id)
    tasks = []

    for item in items:
        if slack_tools.extract_completed(item, schema):
            logger.info("follow_up_skipped_completed task_id=%s",
                        slack_tools.extract_item_id(item) or "unknown")
            continue
        raw_due = slack_tools.extract_due_date(item, schema)
        try:
            due = date.fromisoformat(str(raw_due)[:10])
        except (TypeError, ValueError):
            continue
        item_id = slack_tools.extract_item_id(item)
        if not item_id:
            continue
        owner_ids = slack_tools.extract_assignee_ids(item, schema)
        recipients = [(owner_id, user_name(owner_id)) for owner_id in owner_ids]
        if not recipients and settings.fallback_channel and settings.fallback_actor_id:
            fallback_ctx = config.build_context(
                settings.fallback_actor_id, settings.fallback_channel)
            fallback_ctx.list_id = list_id
            if (config.has_permission(fallback_ctx, "view_others")
                    and all(config.can_read_field(fallback_ctx, field)
                            for field in ("name", "assignee", "due_date"))):
                recipients = [(f"channel:{settings.fallback_channel}", "Unassigned")]
        if not recipients:
            logger.info("follow_up_skipped_unassigned task_id=%s", item_id)
        for owner_id, owner_label in recipients:
            if owner_id.startswith("channel:"):
                tasks.append(deadline_reminders.ReminderTask(
                    task_id=str(item_id),
                    name=slack_tools.extract_item_name(item, schema) or "Unnamed task",
                    owner_id=owner_id,
                    owner_name=owner_label,
                    due_date=due,
                    priority=slack_tools.extract_priority(item, schema),
                    completed=False,
                    status=slack_tools.extract_status(item, schema),
                    timezone=resolve_follow_up_timezone(None, settings)[0],
                ))
                continue
            ctx = config.build_context(owner_id, config.SLACK_LIST_CHANNEL_ID)
            ctx.list_id = list_id
            if not config.has_permission(ctx, "view"):
                continue
            if not all(config.can_read_field(ctx, field) for field in ("name", "assignee", "due_date")):
                continue
            tasks.append(deadline_reminders.ReminderTask(
                task_id=str(item_id),
                name=slack_tools.extract_item_name(item, schema) or "Unnamed task",
                owner_id=owner_id,
                owner_name=owner_label,
                due_date=due,
                priority=(slack_tools.extract_priority(item, schema)
                          if config.can_read_field(ctx, "priority") else None),
                completed=False,
                status=(slack_tools.extract_status(item, schema)
                        if config.can_read_field(ctx, "status") else None),
                timezone=resolve_follow_up_timezone(owner_id, settings)[0],
            ))
    return tasks


def _send_deadline_reminder(recipient_id, message):
    """Send once through chat.postMessage and require Slack delivery evidence."""
    target_type = "configured_channel" if recipient_id.startswith("channel:") else "task_owner"
    target = recipient_id.split(":", 1)[1] if recipient_id.startswith("channel:") else recipient_id
    try:
        response = app.client.chat_postMessage(channel=target, text=slack_mrkdwn(message))
        response_data = slack_tools.data_of(response)
        if not response_data.get("ok", True):
            logger.error("follow_up_failed stage=chat_post_message target_type=%s slack_error=%s",
                         target_type, response_data.get("error", "unknown_error"))
        data = slack_tools.checked(response, "Send follow-up notification")
    except Exception as exc:
        slack_response = getattr(exc, "response", None)
        slack_error = None
        if slack_response is not None:
            try:
                slack_error = slack_tools.data_of(slack_response).get("error")
            except Exception:
                slack_error = None
        logger.error("follow_up_failed stage=chat_post_message target_type=%s error_type=%s slack_error=%s",
                     target_type, type(exc).__name__, slack_error or "unavailable")
        raise
    if not data.get("ts"):
        logger.error("follow_up_failed stage=delivery_confirmation target_type=%s reason=missing_ts",
                     target_type)
        raise RuntimeError("Slack did not confirm follow-up delivery")
    logger.info("follow_up_delivery_confirmed target_type=%s channel_type=%s",
                target_type, "dm" if str(data.get("channel", "")).startswith("D") else "channel")
    return data


def create_reminder_scheduler(settings=None, state_db=None):
    settings = settings or deadline_reminders.ReminderSettings.from_env()
    weekly_settings = deadline_reminders.WeeklySummarySettings.from_env()
    state_path = state_db or DB_PATH
    sentinel_engine = action_item_sentinel.ActionItemSentinel(
        action_item_sentinel.SentinelStore(state_path))

    def build_weekly(period_start, period_end):
        if not weekly_settings.channel or not weekly_settings.actor_id:
            return None
        ctx = config.build_context(
            weekly_settings.actor_id, weekly_settings.channel)
        message = handle_weekly_summary(
            {"intent": "weekly_summary"}, ctx, None,
            period_start=period_start, period_end=period_end)
        return deadline_reminders.WeeklySummaryDelivery(
            list_id=ctx.list_id, channel_id=weekly_settings.channel,
            period_start=period_start, period_end=period_end, message=message)

    return deadline_reminders.DeadlineReminderScheduler(
        settings=settings,
        store=deadline_reminders.ReminderStore(state_path),
        load_tasks=lambda: _load_deadline_reminder_tasks(settings, sentinel_engine),
        send=_send_deadline_reminder,
        weekly_settings=weekly_settings,
        build_weekly=build_weekly,
        send_weekly=lambda channel, message: post(channel, message),
    )


def fmt_task(item, schema, index, ctx=None):
    readable = lambda field: ctx is None or config.can_read_field(ctx, field)
    name = slack_tools.extract_item_name(item, schema) if readable("name") else "Restricted task"
    row = slack_presentation.TaskRow(
        name=name or "Unnamed task",
        assignee=(item_assignees(item, schema) or "Unassigned") if readable("assignee") else None,
        due_date=slack_tools.extract_due_date(item, schema) if readable("due_date") else None,
        show_due=readable("due_date"),
        priority=(slack_tools.extract_priority(item, schema) or "No priority") if readable("priority") else None,
        status=("Completed" if slack_tools.extract_completed(item, schema) else "Pending")
        if readable("status") or readable("completed") else None,
        completed_date=(progress_engine.completion_date(item, schema).isoformat()
                        if readable("status") and progress_engine.completion_date(item, schema)
                        else None),
    )
    return slack_presentation.task_line(row, position=index, today=current_date())


def _ambiguity_choice(item, schema, index, ctx=None):
    """Render one compact candidate without leaking fields hidden by RBAC."""
    readable = lambda field: ctx is None or config.can_read_field(ctx, field)
    name = ((slack_tools.extract_item_name(item, schema) or "Unnamed task")
            if readable("name") else "Restricted task")
    details = []
    if readable("priority"):
        details.append(slack_tools.extract_priority(item, schema) or "No priority")
    if readable("assignee"):
        details.append(item_assignees(item, schema) or "Unassigned")
    if readable("due_date"):
        details.append(slack_presentation.compact_date(
            slack_tools.extract_due_date(item, schema), current_date()))
    if readable("status") or readable("completed"):
        details.append("Completed" if slack_tools.extract_completed(item, schema) else "Pending")
    suffix = " · ".join(slack_presentation.text(value) for value in details)
    return f"{index}. *{slack_presentation.text(name)}*" + (f"\n   {suffix}" if suffix else "")


def _task_rows(items, schema, ctx=None):
    rows = []
    readable = lambda field: ctx is None or config.can_read_field(ctx, field)
    for item in items:
        rows.append(slack_presentation.TaskRow(
            name=(slack_tools.extract_item_name(item, schema) or "Unnamed task") if readable("name") else "Restricted task",
            assignee=(item_assignees(item, schema) or "Unassigned") if readable("assignee") else None,
            due_date=slack_tools.extract_due_date(item, schema) if readable("due_date") else None,
            show_due=readable("due_date"),
            priority=(slack_tools.extract_priority(item, schema) or "No priority") if readable("priority") else None,
            status=("Completed" if slack_tools.extract_completed(item, schema) else "Pending")
            if readable("status") or readable("completed") else None,
            completed_date=(progress_engine.completion_date(item, schema).isoformat()
                            if readable("status") and progress_engine.completion_date(item, schema)
                            else None),
        ))
    return rows


def format_items(items, schema, title="Action Items", ctx=None):
    normalized = str(title or "").casefold()
    if normalized.startswith(("your pending", "my pending", "my action")):
        empty = "You have no pending tasks."
    elif "focus today" in normalized or "today's focus" in normalized:
        empty = "No action items are due today."
    else:
        empty = "No authorized action items found."
    return slack_presentation.task_collection(
        _task_rows(items, schema, ctx), title, empty_message=empty, today=current_date())


def format_created_items(items, schema, ctx=None):
    return slack_presentation.created_collection(
        _task_rows(items, schema, ctx), today=current_date())


def format_status_sections(items, schema, title="Action Items", ctx=None):
    rows = _task_rows(items, schema, ctx)
    return slack_presentation.task_collection(
        rows, title, group_status=True, group_due=True, today=current_date(),
        empty_message="No authorized action items found.")


def make_context_keys(channel_id, thread_ts=None, msg_ts=None, user_id=None, bot_ts=None,
                      team_id=None, list_id=None):
    """Versioned keys isolate workspace, list, channel and requesting user.

    A thread never falls back to channel-root state. Bot-message aliases carry a
    snapshot of the view posted at that timestamp, so replying to an older bot
    message starts from that view rather than the latest channel view.
    """
    scope = ["v2", team_id or "single-workspace", list_id or config.get_list_id_for_channel(channel_id, team_id), channel_id, user_id]
    anchors = []
    if msg_ts:
        anchors.append(msg_ts)
    if thread_ts:
        anchors.append(thread_ts)
    elif not bot_ts:
        anchors.append("root")
    if bot_ts:
        anchors.append(bot_ts)
    return [json.dumps(scope + [anchor], separators=(",", ":")) for anchor in dict.fromkeys(anchors)]


def context_keys(ctx):
    return make_context_keys(ctx.channel_id, ctx.thread_ts, ctx.msg_ts, ctx.user_id,
                             team_id=ctx.team_id, list_id=ctx.list_id)


def _save_state(ctx, entry):
    entry = deepcopy(entry)
    entry.update(channel_id=ctx.channel_id, user_id=ctx.user_id, team_id=ctx.team_id,
                 list_id=ctx.list_id, thread_ts=ctx.thread_ts, msg_ts=ctx.msg_ts,
                 created=time.time())
    for key in context_keys(ctx):
        _ctx_write(key, entry)
    return entry


def store_view(primary_key, items, schema=None, ctx=None, query_filter=None, secondary_keys=None):
    """Store an immutable ordered display snapshot; live changes never renumber it."""
    entry = {
        "items": deepcopy(list(items)),
        "displayed_tasks": [
            {"position": i, "item_id": slack_tools.extract_item_id(item),
             "name": slack_tools.extract_item_name(item, schema), "raw_item": deepcopy(item)}
            for i, item in enumerate(items, 1)
        ] if schema else [],
        "focus_ids": [slack_tools.extract_item_id(item) for item in items] if len(items) == 1 else [],
        "last_task_id": slack_tools.extract_item_id(items[0]) if len(items) == 1 else None,
        "last_task_name": slack_tools.extract_item_name(items[0], schema)
        if schema and len(items) == 1 else None,
        "query_filter": query_filter, "created": time.time(),
    }
    if ctx:
        _save_state(ctx, entry)
    else:
        _ctx_write(primary_key, entry)


def record_bot_response(channel_id, bot_ts, thread_ts=None, msg_ts=None, user_id=None,
                        team_id=None, list_id=None):
    if not bot_ts:
        return
    # Exact incoming message snapshot takes precedence over later conversation state.
    entry = get_thread_context(channel_id, thread_ts, msg_ts, user_id,
                               team_id=team_id, list_id=list_id)
    if entry:
        entry = deepcopy(entry)
        entry["bot_ts"] = bot_ts
        keys = make_context_keys(channel_id, user_id=user_id, bot_ts=bot_ts,
                                 team_id=team_id, list_id=list_id)
        for key in keys:
            _ctx_write(key, entry)


def get_thread_context(channel_id, thread_ts=None, msg_ts=None, user_id=None, primary_key=None,
                       team_id=None, list_id=None):
    keys = make_context_keys(channel_id, thread_ts, msg_ts, user_id,
                             team_id=team_id, list_id=list_id)
    # A caller-supplied key cannot bypass scope isolation.
    if primary_key in keys:
        keys.remove(primary_key)
        keys.insert(0, primary_key)
    for key in keys:
        entry = _ctx_read(key)
        if entry:
            return deepcopy(entry)
    return None


def _state(ctx):
    return get_thread_context(ctx.channel_id, ctx.thread_ts, ctx.msg_ts, ctx.user_id,
                              team_id=ctx.team_id, list_id=ctx.list_id) or {}


def previous_items(key, ctx=None):
    return list((_state(ctx) if ctx else _ctx_read(key)).get("items", []))


def selection_from(parsed, items, schema):
    ref = reference_from(parsed)
    if not ref:
        return []
    ids = select_ids(ref, [slack_tools.extract_item_id(x) for x in items])
    by_id = {slack_tools.extract_item_id(x): x for x in items}
    return [by_id[item_id] for item_id in ids]


def _live_targets(ids, items):
    by_id = {slack_tools.extract_item_id(item): item for item in items}
    if not ids or any(not item_id or item_id not in by_id for item_id in ids):
        raise ValueError("A selected task no longer exists in this Slack List. Please display tasks again; displayed positions have not been shifted.")
    return [by_id[item_id] for item_id in dict.fromkeys(ids)]


def _context_assignees(state):
    query = state.get("query_filter") or {}
    if query.get("resolved_assignee_ids"):
        return list(query["resolved_assignee_ids"])
    values = list(query.get("assignees") or [])
    if query.get("assignee"):
        values.append(query["assignee"])
    return list(dict.fromkeys(values))


def _resolve_assignee_ids(parsed, ctx, state=None):
    if parsed.get("assignee_self"):
        return [ctx.user_id]
    if parsed.get("resolved_assignee_ids"):
        return list(dict.fromkeys(parsed["resolved_assignee_ids"]))
    values = list(parsed.get("assignees") or [])
    if parsed.get("assignee") and parsed["assignee"] not in values:
        values.append(parsed["assignee"])
    if parsed.get("assignee_reference") == "context":
        values.extend(_context_assignees(state or _state(ctx)))
        if not values:
            raise ValueError("I couldn't determine which users you mean from this thread. Please name or mention them.")
    result = []
    for value in values:
        user_id = slack_tools.find_user_id(value)
        if not user_id:
            raise ValueError(f"I couldn't resolve the Slack user {value!r}. Please use an @mention or display name.")
        result.append(user_id)
    return list(dict.fromkeys(result))


def _set_path(value, path, replacement):
    node = value
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = replacement


def _is_requester_reference(value):
    return isinstance(value, str) and value.strip().casefold() in {
        "i", "me", "my", "mine", "myself", "the requester",
    }


def _resolve_command_members(parsed, ctx, path=(), root=None, member_lookup=None):
    """Resolve member text once and retain Slack IDs in the trusted command."""
    root = parsed if root is None else root
    member_lookup = {"records": None} if member_lookup is None else member_lookup

    def candidates(value):
        raw = str(value or "").strip()
        if re.search(r"<@([UW][A-Z0-9]+)(?:\|[^>]+)?>", raw, re.I) or re.fullmatch(
                r"[UW][A-Z0-9]+", raw, re.I):
            return slack_tools.find_user_candidates(value, members=[])
        if member_lookup["records"] is None:
            member_lookup["records"] = slack_tools.workspace_member_records()
        return slack_tools.find_user_candidates(value, members=member_lookup["records"])

    parsed["actor_id"] = ctx.user_id
    if parsed.get("intent") == "compound":
        for index, operation in enumerate(parsed.get("operations") or []):
            _resolve_command_members(
                operation, ctx, path + ("operations", index), root, member_lookup)
        return parsed
    if parsed.get("assignee_self"):
        parsed["resolved_assignee_ids"] = [ctx.user_id]
    member_values = list(parsed.get("members") or [])
    if parsed.get("member") and parsed["member"] not in member_values:
        member_values.append(parsed["member"])
    if parsed.get("member_self"):
        parsed["resolved_member_ids"] = [ctx.user_id]
    elif member_values:
        parsed["members"] = member_values
        member_ids = []
        for index, value in enumerate(member_values):
            if _is_requester_reference(value):
                member_ids.append(ctx.user_id)
                continue
            matches = candidates(value)
            if not matches:
                raise ValueError(f"I couldn't resolve the Slack member {value!r}. Please use an @mention or display name.")
            if len(matches) > 1:
                state = _state(ctx)
                state.update(member_candidates=matches, member_pending={"command": deepcopy(root), "path": list(path) + ["members", index]})
                _save_state(ctx, state)
                choices = "\n".join(f"{i}. {m['label']} (<@{m['id']}>)" for i, m in enumerate(matches, 1))
                raise ValueError(f"Which Slack member do you mean by {value!r}?\n{choices}")
            member_ids.append(matches[0]["id"])
        parsed["resolved_member_ids"] = list(dict.fromkeys(member_ids))
    values = list(parsed.get("assignees") or [])
    if parsed.get("assignee") and parsed["assignee"] not in values:
        values.append(parsed["assignee"])
    if values:
        parsed["assignees"] = values
        resolved = []
        for index, value in enumerate(values):
            if _is_requester_reference(value):
                resolved.append(ctx.user_id)
                continue
            matches = candidates(value)
            if not matches:
                if parsed.get("assignee_tentative") and parsed.get("fallback_task_name"):
                    parsed["task_name"] = parsed.pop("fallback_task_name")
                    parsed["assignees"] = []
                    parsed.pop("assignee_tentative", None)
                    parsed.pop("resolved_assignee_ids", None)
                    parsed.pop("reference", None)
                    parsed.pop("reference_scope", None)
                    parsed.pop("target_selection", None)
                    parsed["target_scope"] = "single"
                    break
                raise ValueError(f"I couldn't resolve the Slack user {value!r}. Please use an @mention or display name.")
            if len(matches) > 1:
                state = _state(ctx)
                state.update(member_candidates=matches, member_pending={"command": deepcopy(root), "path": list(path) + ["assignees", index]})
                _save_state(ctx, state)
                choices = "\n".join(f"{i}. {m['label']} (<@{m['id']}>)" for i, m in enumerate(matches, 1))
                raise ValueError(f"Which Slack member do you mean by {value!r}?\n{choices}")
            resolved.append(matches[0]["id"])
        parsed["resolved_assignee_ids"] = list(dict.fromkeys(resolved))
    for index, change in enumerate(parsed.get("changes") or []):
        if change.get("field") not in {"assignee", "owner"}:
            continue
        raw_values = change["value"] if isinstance(change["value"], list) else [change["value"]]
        ids = []
        for value_index, value in enumerate(raw_values):
            if _is_requester_reference(value):
                ids.append(ctx.user_id)
                continue
            matches = candidates(value)
            if not matches:
                raise ValueError(f"I couldn't resolve the Slack user {value!r}. Please use an @mention or display name.")
            if len(matches) > 1:
                state = _state(ctx)
                member_path = ["changes", index, "value"] + ([value_index] if isinstance(change["value"], list) else [])
                state.update(member_candidates=matches, member_pending={"command": deepcopy(root), "path": list(path) + member_path})
                _save_state(ctx, state)
                choices = "\n".join(f"{i}. {m['label']} (<@{m['id']}>)" for i, m in enumerate(matches, 1))
                raise ValueError(f"Which Slack member do you mean by {value!r}?\n{choices}")
            ids.append(matches[0]["id"])
        change["value"] = ids if isinstance(change["value"], list) else ids[0]
    for index, task in enumerate(parsed.get("tasks") or []):
        _resolve_command_members(task, ctx, path + ("tasks", index), root, member_lookup)
    return parsed


def _filter_items(items, parsed, schema, assignee_ids=(), default_pending=False):
    """Apply the same structured filters to reads and bulk target resolution."""
    temporal = parsed.get("temporal_filter") or {}
    temporal_field = temporal.get("field")
    if temporal_field == "completed_at" and not any(
            progress_engine.completion_date(item, schema) for item in items):
        raise ValueError(
            "Slack List exposes whether a task is completed, but not a reliable completion timestamp, "
            "so I can't determine which tasks were completed on that date.")

    def item_date(item, field):
        if field == "due_date":
            raw = slack_tools.extract_due_date(item, schema)
        elif field == "completed_at":
            return progress_engine.completion_date(item, schema)
        elif field == "created_at":
            return progress_engine.creation_date(item, schema)
        elif field == "updated_at":
            raw = item.get("updated_timestamp") or item.get("date_updated")
        else:
            return None
        if raw in {None, ""}:
            return None
        try:
            return datetime.fromtimestamp(float(raw), ZoneInfo("Asia/Kathmandu")).date()
        except (TypeError, ValueError, OSError):
            try:
                return date.fromisoformat(str(raw)[:10])
            except ValueError:
                return None

    temporal_date = temporal_from = temporal_to = None
    try:
        temporal_date = date.fromisoformat(temporal["date"]) if temporal.get("date") else None
        temporal_from = date.fromisoformat(temporal["date_from"]) if temporal.get("date_from") else None
        temporal_to = date.fromisoformat(temporal["date_to"]) if temporal.get("date_to") else None
    except (TypeError, ValueError):
        raise ValueError("I couldn't understand the requested temporal condition.")
    statuses = set(parsed.get("statuses") or [])
    completed = parsed.get("completed")
    status = config.normalize_status(parsed.get("status"))
    if status in {"open", "completed"}:
        completed = status == "completed"
    elif (parsed.get("all_tasks") or parsed.get("all")):
        completed = None
    elif statuses:
        completed = None
    elif completed is None and default_pending:
        completed = False

    today = current_date()
    week_end = today + timedelta(days=6 - today.weekday())
    try:
        date_from = date.fromisoformat(parsed["date_from"]) if parsed.get("date_from") else None
        date_to = date.fromisoformat(parsed["date_to"]) if parsed.get("date_to") else None
    except (TypeError, ValueError):
        raise ValueError("I couldn't understand the requested date range.")
    if date_from and date_to and date_from > date_to:
        raise ValueError("The start of the requested date range is after its end.")
    priority = config.normalize_priority(parsed.get("priority"))
    query = (parsed.get("query") or "").strip().casefold()
    query_terms = [str(term).strip().casefold() for term in parsed.get("query_terms") or []
                   if str(term).strip()]
    assignee_ids = set(assignee_ids)
    assignee_condition = parsed.get("assignee_condition")
    actor_id = parsed.get("actor_id")
    if assignee_condition in {"self", "other"} and not actor_id:
        raise ValueError("I couldn't determine the requesting Slack user for that assignee filter.")
    filtered = []
    for item in items:
        if statuses:
            item_status = "completed" if slack_tools.extract_completed(item, schema) else "open"
            if item_status not in statuses:
                continue
        if completed is not None and slack_tools.extract_completed(item, schema) != completed:
            continue
        item_assignees = set(slack_tools.extract_assignee_ids(item, schema))
        if assignee_ids and not (item_assignees & assignee_ids):
            continue
        if assignee_condition == "self" and actor_id not in item_assignees:
            continue
        if assignee_condition == "other" and (not item_assignees or actor_id in item_assignees):
            continue
        if assignee_condition == "unassigned" and item_assignees:
            continue
        if assignee_condition == "assigned" and not item_assignees:
            continue
        if priority and slack_tools.extract_priority(item, schema) != priority:
            continue
        item_name = slack_tools.extract_item_name(item, schema).casefold()
        if query_terms and not all(term in item_name for term in query_terms):
            continue
        if query and not query_terms and query not in item_name:
            continue
        due = slack_tools.extract_due_date(item, schema)
        if parsed.get("due_today") and due != today.isoformat():
            continue
        if parsed.get("overdue") and (not due or due >= today.isoformat() or slack_tools.extract_completed(item, schema)):
            continue
        if parsed.get("due_this_week"):
            try:
                due_value = date.fromisoformat(due) if due else None
            except ValueError:
                due_value = None
            if due_value is None or not today <= due_value <= week_end:
                continue
        if date_from or date_to:
            try:
                due_value = date.fromisoformat(due) if due else None
            except ValueError:
                due_value = None
            if due_value is None or (date_from and due_value < date_from) or (date_to and due_value > date_to):
                continue
        if temporal_field:
            value = item_date(item, temporal_field)
            relation = temporal.get("relation")
            if value is None:
                continue
            if relation == "on" and value != temporal_date:
                continue
            if relation == "before" and (temporal_date is None or value >= temporal_date):
                continue
            if relation == "after" and (temporal_date is None or value <= temporal_date):
                continue
            if relation == "between" and (
                    temporal_from is None or temporal_to is None or not temporal_from <= value <= temporal_to):
                continue
        filtered.append(item)
    return filtered


def _sort_items(items, parsed, schema):
    """Apply one normalized sort dimension without changing item identity."""
    sort_by = parsed.get("sort_by")
    if not sort_by:
        return list(items)
    reverse = parsed.get("sort_order", "asc") == "desc"

    def value(item):
        if sort_by == "urgency":
            priority = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}.get(
                slack_tools.extract_priority(item, schema), 5)
            raw_due = slack_tools.extract_due_date(item, schema)
            try:
                due = date.fromisoformat(raw_due).toordinal() if raw_due else date.max.toordinal()
            except ValueError:
                due = date.max.toordinal()
            created = item.get("date_created") or item.get("created_timestamp") or 0
            try:
                created = float(created)
            except (TypeError, ValueError):
                created = 0
            return priority, due, created
        if sort_by == "created_at":
            raw = item.get("date_created") or item.get("created_timestamp") or item.get("updated_timestamp")
            try:
                return float(raw) if raw is not None else None
            except (TypeError, ValueError):
                return None
        if sort_by == "due_date":
            raw = slack_tools.extract_due_date(item, schema)
            try:
                return date.fromisoformat(raw) if raw else None
            except ValueError:
                return None
        if sort_by == "priority":
            # P1 is the highest priority, so its natural ascending rank is 1.
            raw = slack_tools.extract_priority(item, schema)
            return {"P1": 1, "P2": 2, "P3": 3, "P4": 4}.get(raw)
        if sort_by == "status":
            return 1 if slack_tools.extract_completed(item, schema) else 0
        if sort_by == "assignee":
            return slack_tools.extract_assignee(item, schema).casefold()
        return slack_tools.extract_item_name(item, schema).casefold()

    present = [item for item in items if value(item) not in {None, ""}]
    missing = [item for item in items if value(item) in {None, ""}]
    present.sort(key=value, reverse=reverse)
    return present + missing


def _apply_target_selection(items, parsed, schema):
    """Order a candidate set, then select; filtering has already happened."""
    selection = parsed.get("target_selection") or {}
    order_by = selection.get("order_by")
    if order_by and order_by != "position":
        direction = selection.get("direction", "asc")
        if order_by == "created_at" and not any(
                item.get("date_created") or item.get("created_timestamp") or item.get("updated_timestamp")
                for item in items):
            # Slack normally supplies creation timestamps. Preserve API order as
            # the deterministic fallback when a test adapter/schema omits them.
            ordered = list(items)
            if direction == "desc":
                ordered.reverse()
        else:
            ordered = _sort_items(items, {"sort_by": order_by, "sort_order": direction}, schema)
    else:
        ordered = _sort_items(items, parsed, schema)
        if order_by == "position" and selection.get("direction") == "desc":
            ordered.reverse()
    mode = selection.get("mode")
    if mode in {"one", "many"}:
        count = selection.get("count") or (1 if mode == "one" else None)
        if not count:
            raise ValueError("Please specify how many tasks to select.")
        return ordered[:count]
    limit = parsed.get("limit")
    return ordered[:limit] if isinstance(limit, int) and limit > 0 else ordered


def _group_key(item, group_by, schema):
    if group_by == "assignee":
        return item_assignees(item, schema) or "Unassigned"
    if group_by == "status":
        return "Completed" if slack_tools.extract_completed(item, schema) else "Pending"
    if group_by == "priority":
        return slack_tools.extract_priority(item, schema) or "No priority"
    if group_by == "due_date":
        return slack_tools.extract_due_date(item, schema) or "No due date"
    raise ValueError("Please specify a supported grouping field.")


def _format_grouped(items, parsed, schema, title, ctx=None):
    groups = {}
    for item in items:
        groups.setdefault(_group_key(item, parsed["group_by"], schema), []).append(item)
    if parsed.get("aggregate") == "count":
        lines = [f"• *{label}*: {len(group)}" for label, group in groups.items()]
        return f"*{title} — Count by {parsed['group_by'].replace('_', ' ').title()}*\n\n" + ("\n".join(lines) or "No matching action items.")
    lines, position = [], 1
    for label, group in groups.items():
        lines.append(f"*{label}*")
        for item in group:
            lines.append(fmt_task(item, schema, position, ctx))
            position += 1
    return f"*{title} — Grouped by {parsed['group_by'].replace('_', ' ').title()}*\n" + ("\n".join(lines) or "No matching action items.")


def resolve_target_set(parsed, items, schema, memory_key, ctx=None, intent=None):
    """Resolve any target shape to an immutable collection of exact Slack IDs."""
    cardinality = requested_cardinality(parsed)
    # target_ids are created internally by clarification, never trusted from a model.
    if parsed.get("target_ids"):
        targets = _live_targets(parsed["target_ids"], items)
        kind = resolved_target_type(cardinality, parsed.get("target_scope"))
        return ResolvedTargetSet(kind, tuple(slack_tools.extract_item_id(x) for x in targets), tuple(targets))
    ref = reference_from(parsed)
    state = _state(ctx) if ctx else _ctx_read(memory_key)
    assignee_ids = _resolve_assignee_ids(parsed, ctx, state) if ctx else []
    if parsed.get("assignee_tentative") and parsed.get("fallback_task_name") and assignee_ids:
        fallback = normalize_task_name(parsed["fallback_task_name"])
        exact_titles = [item for item in items if normalize_task_name(
            slack_tools.extract_item_name(item, schema)) == fallback]
        if exact_titles:
            raise ValueError(
                "That wording could mean an exact task title or a member-filtered position. "
                "Please use an @mention/possessive for the member, or quote the task title; no changes were made.")
    target_groups = parsed.get("target_groups") or []
    if target_groups:
        base_command = {**parsed, "query": "", "target_groups": []}
        candidates = _filter_items(items, base_command, schema, assignee_ids)
        selected_ids = []
        for group in target_groups:
            group_candidates = _filter_items(
                candidates, {**base_command, "query": group["query"]}, schema, ())
            if not group_candidates:
                raise ValueError(f"No Slack List items match the task group {group['query']!r}.")
            reference = reference_from(group)
            if reference:
                group_ids = select_ids(reference, [slack_tools.extract_item_id(item) for item in group_candidates])
            elif len(group_candidates) == 1:
                group_ids = [slack_tools.extract_item_id(group_candidates[0])]
            else:
                raise ValueError(
                    f"The task group {group['query']!r} matches {len(group_candidates)} items. "
                    "Please specify a position or whether you mean all of them.")
            selected_ids.extend(group_ids)
        targets = _live_targets(list(dict.fromkeys(selected_ids)), items)
        return ResolvedTargetSet(TargetType.MULTIPLE_ITEMS, tuple(dict.fromkeys(selected_ids)), tuple(targets))
    declared_scope = parsed.get("target_scope")
    if declared_scope in {"all_applicable", "filtered"} and not ref:
        targets = _filter_items(items, parsed, schema, assignee_ids)
        targets = _apply_target_selection(targets, parsed, schema)
        kind = resolved_target_type(cardinality, declared_scope)
        return ResolvedTargetSet(kind, tuple(slack_tools.extract_item_id(x) for x in targets), tuple(targets))
    if declared_scope == "contextual" and not ref:
        targets = list(state.get("candidates") or state.get("items") or [])
        if not targets:
            raise ValueError("I couldn't find a displayed task collection in this thread.")
        targets = _apply_target_selection(targets, parsed, schema)
        live = _live_targets([slack_tools.extract_item_id(x) for x in targets], items)
        return ResolvedTargetSet(resolved_target_type(cardinality, declared_scope),
                                 tuple(slack_tools.extract_item_id(x) for x in live), tuple(live))
    if ref:
        if parsed.get("target_scope") == "all_applicable":
            source = _filter_items(items, parsed, schema, assignee_ids)
            target_type = TargetType.ALL_APPLICABLE_ITEMS
            source_scope = "all_applicable"
        elif parsed.get("reference_scope") == "filtered":
            displayed = state.get("items") or []
            if parsed.get("candidate_source") == "displayed" and displayed:
                source = _filter_items(displayed, parsed, schema, assignee_ids)
                if not source:
                    raise ValueError(
                        "No tasks in the displayed conversation context match those filters; no changes were made.")
            else:
                source = _filter_items(items, parsed, schema, assignee_ids)
            target_type = TargetType.FILTERED_COLLECTION
            source_scope = "filtered"
        else:
            source = state.get("candidates") or state.get("items") or []
            target_type = TargetType.CONTEXTUAL_ITEMS
            source_scope = "contextual"
        if not source and not (ref.kind == "focus" and state.get("focus_ids")):
            raise ValueError("I couldn't find any recently displayed action items in this thread to reference. Please run 'show tasks' first.")
        source = _sort_items(source, parsed, schema)
        ids = select_ids(ref, [slack_tools.extract_item_id(x) for x in source],
                         () if state.get("candidates") else state.get("focus_ids", ()))
        # Apply positions BEFORE looking up live items. Never compact the display.
        targets = _live_targets(ids, items)
        return ResolvedTargetSet(resolved_target_type(cardinality, source_scope),
                                 tuple(ids), tuple(targets))

    name = parsed.get("task_name")
    pool = [x for x in items if set(slack_tools.extract_assignee_ids(x, schema)) & set(assignee_ids)] if assignee_ids else items
    matches = slack_tools.find_matches(pool, name, schema) if name else []
    if intent == "complete" and len(matches) > 1:
        pending = [x for x in matches if not slack_tools.extract_completed(x, schema)]
        if pending:
            matches = pending
    if len(matches) <= 1:
        kind = resolved_target_type(cardinality, parsed.get("target_scope"))
        return ResolvedTargetSet(kind, tuple(slack_tools.extract_item_id(x) for x in matches), tuple(matches))
    state.update(candidates=deepcopy(matches), parsed=deepcopy(parsed), intent=intent)
    if ctx:
        _save_state(ctx, state)
    else:
        _ctx_write(memory_key, state)
    if intent == "source":
        choices = "\n\n".join(_ambiguity_choice(item, schema, index, ctx)
                               for index, item in enumerate(matches, 1))
        raise ValueError(
            f"I found {len(matches)} similar tasks. Which one do you mean?\n\n{choices}\n\n"
            "Reply with the number, an ordinal such as *the first one*, or the exact task name.")
    choices = "\n".join(fmt_task(item, schema, index, ctx)
                        for index, item in enumerate(matches, 1))
    raise ValueError(
        f"I found {len(matches)} tasks matching {name!r}:\n\n{choices}\n\n"
        f"Which {name} task should I use?")


def resolve_targets(parsed, items, schema, memory_key, ctx=None, intent=None):
    """Compatibility wrapper for callers that still need item snapshots."""
    return list(resolve_target_set(parsed, items, schema, memory_key, ctx, intent).items)


def handle_list(parsed, ctx, memory_key):
    if not config.has_permission(ctx, "view"):
        raise PermissionError("Your role cannot view action items.")
    list_id = ctx.list_id
    if not list_id: raise ValueError("This channel is not mapped to a Slack List.")
    schema = slack_tools.get_list_schema(list_id)
    items = slack_tools.list_action_items(ctx, list_id)

    # ── Resolve assignee filter ───────────────────────────────────────────
    assignee_ids = _resolve_assignee_ids(parsed, ctx)
    if not config.has_permission(ctx, "view_others"):
        if parsed.get("assignee_condition") in {"other", "unassigned", "assigned"}:
            raise PermissionError("Your role can only view tasks assigned to you.")
        if assignee_ids and any(user_id != ctx.user_id for user_id in assignee_ids):
            raise PermissionError("Your role can only view tasks assigned to you.")
        assignee_ids = [ctx.user_id]

    # Use the same filter engine as exact-ID bulk target resolution.
    filtered = _filter_items(items, parsed, schema, assignee_ids, default_pending=True)

    # ── Query operation ───────────────────────────────────────────────────
    sort_by = parsed.get("sort_by")
    filtered = _apply_target_selection(filtered, parsed, schema)

    # ── Limit ─────────────────────────────────────────────────────────────
    limit = parsed.get("limit")
    if parsed.get("selection") == "first" and not parsed.get("target_selection"):
        filtered = filtered[:1]
    elif parsed.get("selection") == "both":
        filtered = filtered[:2]
    elif parsed.get("selection_count"):
        filtered = filtered[:int(parsed["selection_count"])]

    # ── Title ─────────────────────────────────────────────────────────────
    is_self = bool(parsed.get("assignee_self"))
    is_named = bool((parsed.get("assignees") or parsed.get("assignee")) and not is_self)
    assignee_label = None
    if is_named:
        assignee_label = ", ".join(user_name(uid) for uid in assignee_ids)
    completed = parsed.get("completed")
    normalized_status = config.normalize_status(parsed.get("status"))
    if len(set(parsed.get("statuses") or [])) > 1:
        completed = None
    elif normalized_status in {"open", "completed"}:
        completed = normalized_status == "completed"
    elif parsed.get("all_tasks") or parsed.get("all"):
        completed = None
    elif completed is None:
        completed = False

    if parsed.get("search"):
        title = "Search Results"
    elif sort_by == "due_date" and limit == 1:
        if is_self:
            title = "Your Next Task"
        elif is_named and assignee_label:
            title = f"{assignee_label}'s Next Task"
        else:
            title = "Next Task by Deadline"
    elif parsed.get("due_today"):
        if is_self:
            title = "Focus Today"
        elif is_named and assignee_label:
            title = f"{assignee_label}'s Due Today"
        else:
            title = "Due Today"
    elif parsed.get("overdue"):
        title = "Overdue Action Items"
    elif completed is False:
        title = "My Action Items" if is_self else "Pending Action Items"
    elif completed is True:
        title = "Completed Action Items"
    elif is_self:
        title = "Your Action Items"
    elif is_named and assignee_label:
        title = f"{assignee_label}'s Action Items"
    else:
        title = "Action Items"

    secondary_keys = make_context_keys(ctx.channel_id, ctx.thread_ts, getattr(ctx, "msg_ts", None), ctx.user_id)
    stored_filter = deepcopy(parsed)
    stored_filter["resolved_assignee_ids"] = assignee_ids
    store_view(memory_key, filtered, schema=schema, ctx=ctx, query_filter=stored_filter, secondary_keys=secondary_keys)

    # Helpful empty message for self-scoped due-today queries
    if not filtered and parsed.get("due_today") and is_self:
        overall_pending = _filter_items(
            items, {"completed": False}, schema, [ctx.user_id], default_pending=True)
        return (f"*🎯 {title}*\n\nNo action items are due today.\n\n"
                f"You have {len(overall_pending)} pending task"
                f"{'s' if len(overall_pending) != 1 else ''} overall.")

    if not filtered and parsed.get("search"):
        return "*🔎 Task Search*\n\nNo matching action items found."

    if not filtered and is_self and completed is False:
        return "*📋 My Action Items*\n\nNo pending tasks found.\n\nYou're all caught up."

    if parsed.get("group_by"):
        return _format_grouped(filtered, parsed, schema, title, ctx)
    if parsed.get("aggregate") == "count" or parsed.get("count_only"):
        return f"{len(filtered)} matching action item(s).\n\n" + format_items(filtered, schema, title, ctx)
    if completed is None and filtered:
        rendered = format_status_sections(filtered, schema, title, ctx)
    else:
        rendered = format_items(filtered, schema, title, ctx)
    if parsed.get("search"):
        rendered += f"\n\n{len(filtered)} matching task{'s' if len(filtered) != 1 else ''}"
    if parsed.get("due_today") and is_self and filtered:
        rendered = rendered.replace(
            f"*{title}* · {len(filtered)}",
            f"*{title}* · {len(filtered)}\nThese are the action items due today:", 1)
    return rendered





def handle_create(parsed, ctx):
    if not config.has_permission(ctx, "create"): raise PermissionError("You do not have permission to create action items.")
    name = (parsed.get("task_name") or "").strip()
    logger.info("task_create_started list_id=%s actor_id=%s title=%r",
                ctx.list_id, ctx.user_id, redact(name))

    assignee_raw = list(parsed.get("resolved_assignee_ids") or parsed.get("assignees") or [])
    if parsed.get("assignee") and parsed["assignee"] not in assignee_raw:
        assignee_raw.append(parsed["assignee"])
    if parsed.get("assignee_self"): assignee_raw = [ctx.user_id]
    if not config.has_permission(ctx, "create_for_others"):
        if assignee_raw and any(value != ctx.user_id for value in assignee_raw):
            raise PermissionError("Your role can only create tasks assigned to you.")
        assignee_raw = [ctx.user_id]
    priority = parsed.get("priority")
    due_date = parsed.get("due_date")
    if priority:
        priority = config.normalize_priority(priority)
        if not priority:
            raise ValueError("Priority must be P1, P2, P3 or P4.")
    if due_date:
        due_date = date.fromisoformat(str(due_date)).isoformat()
        if due_date < current_date().isoformat():
            raise ValueError("The due date cannot be in the past.")
    for field, value in (("priority", priority), ("assignee", assignee_raw), ("due_date", due_date)):
        if value and not config.can_edit_field(ctx, field):
            raise PermissionError(f"Your role cannot set the {field.replace('_', ' ')} field.")

    if not name or name == "__LAST__":
        return slack_presentation.clarification(
            "Please specify the action item name.")

    # Assignee is optional — tasks can be created unassigned
    assignee = []
    if assignee_raw:
        for value in assignee_raw:
            user_id = slack_tools.find_user_id(value)
            if not user_id:
                return slack_presentation.clarification(
                    "I couldn't find that Slack member. Please use an @mention or a valid display name.")
            assignee.append(user_id)
        assignee = list(dict.fromkeys(assignee))
    assignee_value = assignee[0] if len(assignee) == 1 else assignee

    schema = slack_tools.get_list_schema(ctx.list_id)
    items = slack_tools.list_action_items(ctx, ctx.list_id)
    normalized = normalize_task_name(name)

    expected = [{"field": "name", "value": name}, {"field": "completed", "value": False}]
    if priority:
        expected.append({"field": "priority", "value": priority})
    if assignee:
        expected.append({"field": "assignee", "value": assignee_value})
    if due_date:
        expected.append({"field": "due_date", "value": due_date})
    for change in expected:
        slack_tools._write_cell(schema, change["field"], change["value"])

    op_key = delivery.checkpoint_key("create", {"list": ctx.list_id, "assignee": assignee or None, "fields": expected})
    checkpoint = delivery.checkpoint_read(op_key)
    item_id = checkpoint.get("item_id") if checkpoint else None
    exact = [x for x in items if normalize_task_name(slack_tools.extract_item_name(x, schema)) == normalized
             and slack_tools.extract_assignee_ids(x, schema) == assignee
             and not slack_tools.extract_completed(x, schema)]
    if checkpoint and not item_id:
        new_items = [x for x in exact if slack_tools.extract_item_id(x) not in checkpoint["before_ids"]]
        if len(new_items) != 1:
            return "*Creation outcome unknown* — a previous write was interrupted. Please check the Slack List before retrying; I have not created another task."
        item_id = slack_tools.extract_item_id(new_items[0])
    elif not checkpoint and exact:
        if len(exact) > 1:
            _save_state(ctx, {**_state(ctx), "candidates": exact, "parsed": {"intent": "inspect", "task_name": name}, "intent": "inspect"})
            return "Multiple existing tasks have that name and assignee. Which one do you mean?\n" + format_items(exact, schema, ctx=ctx)
        existing = exact[0]
        check = mutations.verify(slack_tools.extract_item_id(existing), expected, ctx, schema)
        store_view(context_keys(ctx)[0], exact, schema, ctx)
        if not check.verified:
            requested_assignees = [user_name(value) for value in assignee]
            requested_row = slack_presentation.TaskRow(
                name=name,
                assignee=", ".join(requested_assignees) if requested_assignees else "Unassigned",
                due_date=due_date,
                show_due=True,
                priority=priority or "No priority",
                status="Pending",
            )
            existing_row = _task_rows(exact, schema, ctx)[0]
            return (slack_presentation.task_conflict(
                        requested_row, existing_row, today=current_date())
                    + "\nAsk me to update it explicitly if you want these values applied.")
        return ("*Task already exists*\n\n"
                + slack_presentation.task_summary(
                    _task_rows(exact, schema, ctx)[0], show_status=False,
                    today=current_date())
                + "\n\nNo changes were made.")
    if not item_id:
        delivery.checkpoint_write(op_key, "started", {"before_ids": [slack_tools.extract_item_id(x) for x in items]})
        try:
            item = slack_tools.create_action_item(name, priority, assignee_value if assignee else None, due_date, ctx, ctx.list_id)
            item_id = slack_tools.extract_item_id(item)
        except Exception:
            # Reconcile a timeout after creation against exact fields and pre-write IDs.
            live = slack_tools.list_action_items(ctx, ctx.list_id)
            old_ids = {slack_tools.extract_item_id(x) for x in items}
            fresh = [x for x in live if slack_tools.extract_item_id(x) not in old_ids
                     and normalize_task_name(slack_tools.extract_item_name(x, schema)) == normalized
                     and slack_tools.extract_assignee_ids(x, schema) == assignee]
            if len(fresh) != 1:
                return "*Creation outcome unknown* — Slack did not confirm the write. Please check the list before retrying."
            item_id = slack_tools.extract_item_id(fresh[0])
        delivery.checkpoint_write(op_key, "written", {"item_id": item_id, "before_ids": [slack_tools.extract_item_id(x) for x in items]})
    verified = mutations.verify(item_id, expected, ctx, schema)
    if not item_id or not verified.verified:
        return "*Creation not verified*\n" + "; ".join(verified.problems or ["Slack returned no item ID"])
    item = verified.item
    _record_verified_audit(ctx, schema, item_id, "create", expected, None, item)
    if parsed.get("_source"):
        try:
            source_trace.record(DB_PATH, ctx, item_id, parsed["_source"])
        except Exception:
            logger.warning("Unable to persist source trace for verified item %s", item_id)
    store_view(context_keys(ctx)[0], [item], schema, ctx)
    task_name_out = slack_tools.extract_item_name(item, schema) if config.can_read_field(ctx, "name") else "Restricted task"
    assignee_out = (item_assignees(item, schema) or "Unassigned") if config.can_read_field(ctx, "assignee") else None
    row = slack_presentation.TaskRow(
        name=task_name_out,
        assignee=assignee_out,
        due_date=due_date if config.can_read_field(ctx, "due_date") else None,
        show_due=config.can_read_field(ctx, "due_date"),
        priority=(priority or "No priority") if config.can_read_field(ctx, "priority") else None,
        status="Pending" if config.can_read_field(ctx, "status") or config.can_read_field(ctx, "completed") else None)
    logger.info("task_verification_completed list_id=%s actor_id=%s item_id=%s verified=true",
                ctx.list_id, ctx.user_id, item_id)
    logger.info("task_create_completed list_id=%s actor_id=%s item_id=%s",
                ctx.list_id, ctx.user_id, item_id)
    return slack_presentation.created_collection([row], today=current_date())



def handle_create_multi(tasks_list: list, ctx) -> str:
    """
    Create multiple action items from a list of per-task parsed dicts.

    Each entry in tasks_list is a dict with the same shape as a single-task
    parse result (task_name, assignee, assignee_self, priority, due_date, ...).

    Returns a combined Slack message summarising all successes and failures.
    Never silently drops a task — every task is attempted and every result is reported.
    """
    successes = []
    failures = []
    created_items = []

    for i, task_parsed in enumerate(tasks_list, 1):
        if not isinstance(task_parsed, dict):
            failures.append(f"{i}. (invalid task entry — skipped)")
            continue

        task_name = (task_parsed.get("task_name") or "").strip()
        if not task_name:
            failures.append(f"{i}. (empty task name — skipped)")
            continue

        # handle_create expects a parsed dict with an 'intent' key
        single = dict(task_parsed)
        single.setdefault("intent", "create")

        try:
            result = handle_create(single, ctx)
            if result:
                normalized_result = result.casefold()
                if "task already exists" in normalized_result:
                    successes.append(result)
                elif "task created" not in normalized_result and "tasks created" not in normalized_result:
                    failures.append(f"{i}. *{task_name}* — {result}")
                if "task created" in normalized_result or "tasks created" in normalized_result:
                    created_items.extend(_state(ctx).get("items", []))
            else:
                failures.append(f"{i}. *{task_name}* — no response returned")
        except (ValueError, PermissionError) as exc:
            failures.append(f"{i}. *{task_name}* — {exc}")
        except Exception:
            logger.error("A batch creation failed before verification")
            failures.append(f"{i}. *{task_name}* — unable to verify creation")

    if created_items:
        schema = slack_tools.get_list_schema(ctx.list_id)
        store_view(context_keys(ctx)[0], created_items, schema, ctx)
    parts = []
    if created_items:
        unique_created = {slack_tools.extract_item_id(item): item for item in created_items}
        created_items = list(unique_created.values())
        parts.append(format_created_items(created_items, schema, ctx))
    if successes:
        parts.append("\n".join(successes))
    if failures:
        failure_block = "*The following tasks could not be created:*\n" + "\n".join(failures)
        parts.append(failure_block)

    return ("\n\n".join(parts) if parts else slack_presentation.failure(
        "No tasks could be processed.",
        next_step="Check the task names and assignees, then try again."))


def _create_assignee_ids(parsed, ctx):
    if parsed.get("assignee_self"):
        return [ctx.user_id]
    values = list(parsed.get("resolved_assignee_ids") or [])
    if not values and not config.has_permission(ctx, "create_for_others"):
        values = [ctx.user_id]
    return list(dict.fromkeys(values))


def maybe_confirm_duplicate_create(parsed, ctx):
    """Stage one exact, time-limited confirmation for strong near-duplicates."""
    if parsed.get("_duplicate_approved"):
        return None
    if not config.has_permission(ctx, "create"):
        raise PermissionError("You do not have permission to create action items.")
    tasks = parsed.get("tasks") or [parsed]
    logger.info("duplicate_check_started list_id=%s actor_id=%s task_count=%d",
                ctx.list_id, ctx.user_id, len(tasks))
    schema = slack_tools.get_list_schema(ctx.list_id)
    items = slack_tools.list_action_items(ctx, ctx.list_id)
    candidates = []
    for index, task in enumerate(tasks, 1):
        name = (task.get("task_name") or "").strip()
        if not name:
            continue
        assignees = _create_assignee_ids(task, ctx)
        logger.info(
            "Duplicate comparison input task_index=%d title=%r assignee_ids=%s "
            "due_date=%s priority=%s status=%s",
            index, redact(name), assignees, task.get("due_date") or "none",
            task.get("priority") or "none", task.get("status") or "pending",
        )
        matches = workflow_safety.likely_duplicates(name, items, schema, assignees)
        # Preserve the established behavior: the same title may legitimately
        # exist for a different assignee, while the same title/assignee is
        # handled idempotently by handle_create.
        exact_existing = [item for item in matches
                          if normalize_task_name(slack_tools.extract_item_name(item, schema)) == normalize_task_name(name)]
        candidates.extend(item for item in matches if item not in exact_existing)
    unique = {slack_tools.extract_item_id(item): item for item in candidates}
    if not unique:
        return None
    logger.info("duplicate_found list_id=%s actor_id=%s candidate_count=%d",
                ctx.list_id, ctx.user_id, len(unique))
    ids = list(unique)
    state = _state(ctx)
    state["confirmation"] = {
        "kind": "duplicate_create", "created": time.time(),
        "expires_at": time.time() + workflow_safety.CONFIRMATION_TTL_SECONDS,
        "command": deepcopy(parsed), "item_ids": ids,
        "fingerprint": workflow_safety.snapshot_fingerprint(items, schema, ids),
    }
    _save_state(ctx, state)
    return ("*⚠️ Possible Duplicate*\n\n" + format_items(list(unique.values()), schema, "Existing similar tasks", ctx)
            + "\n\nReply *confirm* within 10 minutes to create another task, or *cancel*.")


def _mutation_field_value(item, field, schema):
    if field == "name":
        return slack_tools.extract_item_name(item, schema)
    if field in {"assignee", "owner"}:
        return item_assignees(item, schema) or "Unassigned"
    if field == "due_date":
        return slack_presentation.compact_date(slack_tools.extract_due_date(item, schema))
    if field == "priority":
        return slack_tools.extract_priority(item, schema) or "No priority"
    if field in {"status", "completed"}:
        return "Completed" if slack_tools.extract_completed(item, schema) else "Pending"
    value = slack_tools.extract_field_value(item, schema, field)
    return str(value) if value not in {None, ""} else "Not set"


def _verified_mutation_block(before, after, changes, schema, ctx):
    row = _task_rows([after], schema, ctx)[0]
    completed_change = any(change.get("field") in {"status", "completed"}
                           for change in changes)
    lines = [slack_presentation.task_summary(
        row, show_due=not completed_change, show_status=False, today=current_date())]
    labels = {"name": "Task", "assignee": "Assignee", "owner": "Assignee",
              "due_date": "Due", "priority": "Priority", "status": "Status",
              "completed": "Status"}
    for change in changes:
        field = change.get("field")
        if not field or not config.can_read_field(ctx, field):
            continue
        earlier = _mutation_field_value(before, field, schema)
        later = _mutation_field_value(after, field, schema)
        label = labels.get(field, str(field).replace("_", " ").title())
        lines.append(f"  • {label}: {slack_presentation.text(earlier)} → {slack_presentation.text(later)}")
    if len(lines) == 1:
        lines.append("  • Verified")
    return "\n".join(lines)


def _proposed_change_value(change):
    field, value = change.get("field"), change.get("value")
    if field == "due_date":
        return slack_presentation.compact_date(value, current_date())
    if field == "priority":
        return value or "No priority"
    if field in {"assignee", "owner"}:
        values = value if isinstance(value, list) else [value]
        return ", ".join(user_name(user_id) for user_id in values) or "Unassigned"
    if field in {"completed", "status"}:
        return "Completed" if value in {True, "completed"} else "Pending"
    return str(value)


def _bulk_mutation_preview(items, item_ids, excluded, changes, intent, schema, ctx):
    """Render the exact authorized proposal without changing Slack List state."""
    by_id = {slack_tools.extract_item_id(item): item for item in items}
    labels = {"name": "Task", "assignee": "Owner", "owner": "Owner",
              "due_date": "Due", "priority": "Priority", "status": "Status",
              "completed": "Status"}
    lines = ["*🧠 Proposed Changes — Confirmation Required*", "",
             f"{len(item_ids)} task{'s' if len(item_ids) != 1 else ''} will be updated:"]
    for index, item_id in enumerate(item_ids, 1):
        item = by_id[item_id]
        name = slack_tools.extract_item_name(item, schema) or "Unnamed task"
        owner = item_assignees(item, schema) or "Unassigned"
        lines.extend(("", f"{index}. *{slack_presentation.text(name)}* · {slack_presentation.text(owner)}"))
        for change in changes:
            field = change["field"]
            earlier = _mutation_field_value(item, field, schema)
            later = _proposed_change_value(change)
            lines.append(
                f"   {labels.get(field, field.replace('_', ' ').title())}: "
                f"{slack_presentation.text(earlier)} → {slack_presentation.text(later)}")
    if excluded:
        lines.extend(("", "*Excluded:*"))
        for item_id, reason in excluded:
            item = by_id.get(item_id)
            name = slack_tools.extract_item_name(item, schema) if item else "Unavailable task"
            lines.append(f"• {slack_presentation.text(name)} — {slack_presentation.text(reason)}")
    lines.extend(("", "Apply these changes?", "Reply *confirm*, *yes*, *apply*, or *do it*. Reply *cancel* to stop."))
    return "\n".join(lines)



def handle_mutation(parsed, ctx, memory_key):
    tasks = parsed.get("tasks") or []
    if tasks:
        results = []
        for task in tasks:
            sub = {k: v for k, v in parsed.items() if k != "tasks"}
            sub.update(task if isinstance(task, dict) else {"task_name": task})
            try:
                results.append(handle_mutation(sub, ctx, memory_key))
            except (ValueError, PermissionError) as exc:
                results.append(f"• {sub.get('task_name') or 'Task'}: {exc}")
        return "\n\n".join(results)

    intent = parsed["intent"]
    schema = slack_tools.get_list_schema(ctx.list_id)
    # Validate permissions and all requested fields before any writes.
    changes = mutations.prepare_changes(parsed, ctx, schema, current_date())
    items = slack_tools.list_action_items(ctx, ctx.list_id)
    def build_plan():
        if parsed.get("bulk_preview_required"):
            assignee_ids = _resolve_assignee_ids(parsed, ctx)
            selected = _filter_items(
                items, parsed, schema, assignee_ids, default_pending=False)
            return {
                "target_type": TargetType.FILTERED_COLLECTION.value,
                "item_ids": [slack_tools.extract_item_id(item) for item in selected],
                "changes": changes,
            }
        resolved = resolve_target_set(parsed, items, schema, memory_key, ctx, intent)
        return {"target_type": resolved.target_type.value, "item_ids": list(resolved.item_ids), "changes": changes}
    plan = delivery.stable_plan({"command": parsed, "list": ctx.list_id}, build_plan)
    item_ids, changes = plan["item_ids"], plan["changes"]
    if not item_ids:
        if plan["target_type"] in {TargetType.FILTERED_COLLECTION.value, TargetType.ALL_APPLICABLE_ITEMS.value}:
            raise ValueError("No Slack List items match that collection and its filters.")
        raise ValueError("I couldn't identify the action item or collection. Please clarify the intended target.")
    # Final mutation-boundary invariant: resolution may narrow a candidate
    # collection, but it may never expand a singular request into bulk writes.
    enforce_cardinality(parsed, item_ids)
    if len(item_ids) > 1 and not config.has_permission(ctx, "bulk"):
        raise PermissionError("Your role cannot perform bulk action-item operations.")
    authorized, excluded = [], []
    for item_id in item_ids:
        try:
            mutations.authorize_collection([item_id], items, intent, changes, ctx, schema)
            authorized.append(item_id)
        except PermissionError as exc:
            excluded.append((item_id, str(exc)))
    if not authorized:
        raise PermissionError(excluded[0][1] if excluded else "No action items are authorized for this operation.")
    item_ids = authorized
    requires_preview = bool(parsed.get("bulk_preview_required")) or workflow_safety.confirmation_required(
        intent, len(item_ids), changes, config.CONFIRMATION_THRESHOLD)
    if not parsed.get("_confirmation_approved") and requires_preview:
        state = _state(ctx)
        state["confirmation"] = {
            "kind": "bulk_mutation", "created": time.time(),
            "expires_at": time.time() + workflow_safety.CONFIRMATION_TTL_SECONDS,
            "command": deepcopy(parsed), "item_ids": list(item_ids),
            "fingerprint": workflow_safety.snapshot_fingerprint(items, schema, item_ids),
            "item_fingerprints": {
                item_id: workflow_safety.item_fingerprint(
                    next(item for item in items if slack_tools.extract_item_id(item) == item_id), schema)
                for item_id in item_ids
            },
        }
        _save_state(ctx, state)
        return _bulk_mutation_preview(items, item_ids, excluded, changes, intent, schema, ctx)
    # The mutation boundary consumes exact IDs only; it never performs name matching.
    results = mutations.execute_collection(item_ids, intent, changes, ctx, schema)
    names_by_id = {slack_tools.extract_item_id(item): slack_tools.extract_item_name(item, schema) for item in items}
    before_by_id = {slack_tools.extract_item_id(item): item for item in items}
    state = _state(ctx)
    state.pop("candidates", None)
    state.pop("parsed", None)
    state.pop("intent", None)
    # Preserve the original display, while pronouns can refer to the explicitly selected task.
    state["focus_ids"] = list(item_ids)
    _save_state(ctx, state)
    verb = {"complete": "completed", "reopen": "reopened", "delete": "deleted", "update": "updated"}[intent]
    # Fail closed if a result object and its re-fetched item snapshot disagree.
    # This protects response generation even if a future adapter regression
    # incorrectly marks an inconsistent verification result as successful.
    for result in results:
        if result.verified and intent == "complete" and (
                result.item is None or not slack_tools.extract_completed(result.item, schema)):
            result.verified = False
            result.outcome = "failed"
            result.problems.append("task status in Slack List still shows pending")
        elif result.verified and intent == "reopen" and (
                result.item is None or slack_tools.extract_completed(result.item, schema)):
            result.verified = False
            result.outcome = "failed"
            result.problems.append("task status in Slack List still shows completed")
    for item_id, result in zip(item_ids, results):
        if result.verified:
            _record_verified_audit(
                ctx, schema, item_id, intent, changes, before_by_id.get(item_id),
                None if intent == "delete" else result.item)
    all_verified = all(result.verified for result in results)
    if all_verified:
        heading = (f"✓ Action item {verb}" if len(item_ids) == 1
                   else f"✓ Action items {verb} · {len(item_ids)}")
    else:
        heading = "Action results — not all changes verified"
    lines = []
    for item_id, result in zip(item_ids, results):
        name = names_by_id.get(item_id) or "Action item"
        if not result.verified:
            lines.append(f"• *{name}*: " + "; ".join(result.problems))
        elif intent == "delete":
            lines.append(f"• *{name}* — deletion verified")
        elif len(item_ids) == 1:
            lines.append(_verified_mutation_block(
                before_by_id.get(item_id) or {}, result.item, changes, schema, ctx))
        else:
            # Bullets avoid introducing a second, conflicting set of displayed positions.
            lines.append(re.sub(r"^1\. ", "• ", fmt_task(result.item, schema, 1, ctx)))
    response = f"*{heading}*\n\n" + "\n".join(lines)
    if all_verified:
        response += "\n\nVerified in *Action Items*."
    return response


def handle_inspect(parsed, ctx, memory_key):
    if not config.has_permission(ctx, "view"):
        raise PermissionError("Your role cannot view action items.")
    schema = slack_tools.get_list_schema(ctx.list_id)
    items = slack_tools.list_action_items(ctx, ctx.list_id)
    targets = resolve_targets(parsed, items, schema, memory_key, ctx, "inspect")
    if not targets:
        return "I couldn't find that action item. Please specify its name or displayed number."
    if not config.has_permission(ctx, "view_others") and any(
            ctx.user_id not in slack_tools.extract_assignee_ids(item, schema) for item in targets):
        raise PermissionError("Your role can only view tasks assigned to you.")
    state = _state(ctx)
    state.pop("candidates", None)
    state.pop("parsed", None)
    state["focus_ids"] = [slack_tools.extract_item_id(x) for x in targets]
    _save_state(ctx, state)
    # An information answer changes focus but does not renumber the original view.
    return "\n".join(re.sub(r"^1\. ", "• ", fmt_task(x, schema, 1, ctx)) for x in targets)


def handle_source(parsed, ctx, memory_key):
    if not config.has_permission(ctx, "view"):
        raise PermissionError("Your role cannot view action-item sources.")
    schema = slack_tools.get_list_schema(ctx.list_id)
    items = slack_tools.list_action_items(ctx, ctx.list_id)
    targets = resolve_targets(parsed, items, schema, memory_key, ctx, "source")
    if len(targets) != 1:
        raise ValueError("Please identify one exact task whose source you want to inspect.")
    item = targets[0]
    if (not config.has_permission(ctx, "view_others")
            and ctx.user_id not in slack_tools.extract_assignee_ids(item, schema)):
        raise PermissionError("Your role can only view tasks assigned to you.")
    item_id = slack_tools.extract_item_id(item)
    source = source_trace.get(DB_PATH, ctx.list_id, item_id)
    name = slack_tools.extract_item_name(item, schema)
    if not source:
        return f"No source context is available for *{slack_presentation.text(name)}*."
    source_label = {
        "slack_message": "Slack message", "slack_thread": "Slack conversation",
        "audio": "Audio", "video": "Video", "transcript": "Transcript",
    }.get(source["source_type"], str(source["source_type"]).replace("_", " ").title())
    response = (f"*🧠 Task Context · Source for {slack_presentation.text(name)}*\n\n"
                f"*Source:* {slack_presentation.text(source_label)}")
    if source.get("evidence"):
        response += f"\n*Created from:* “{slack_presentation.text(source['evidence'])}”"
    if config.can_read_field(ctx, "assignee"):
        response += f"\n*Owner:* {slack_presentation.text(item_assignees(item, schema) or 'Unassigned')}"
    if config.can_read_field(ctx, "due_date"):
        due = slack_presentation.compact_date(
            slack_tools.extract_due_date(item, schema), current_date())
        response += f"\n*Due:* {slack_presentation.text(due)}"
    if config.can_read_field(ctx, "priority"):
        priority = slack_tools.extract_priority(item, schema) or "No priority"
        response += f"\n*Priority:* {slack_presentation.text(priority)}"
    return response


def handle_progress(parsed, ctx, memory_key):
    """Calculate read-only analytics from the current authorized List snapshot."""
    if not config.has_permission(ctx, "view"):
        raise PermissionError("Your role cannot view action-item progress.")
    if not ctx.list_id:
        raise ValueError("This channel is not mapped to a Slack List.")
    schema = slack_tools.get_list_schema(ctx.list_id)
    items = slack_tools.list_action_items(ctx, ctx.list_id)
    if (parsed.get("assignee") or parsed.get("assignees") or parsed.get("assignee_self")
            or parsed.get("assignee_condition")) and not config.can_read_field(ctx, "assignee"):
        raise PermissionError("Your role cannot use assignee data for progress reporting.")
    if parsed.get("priority") and not config.can_read_field(ctx, "priority"):
        raise PermissionError("Your role cannot use priority data for progress reporting.")
    assignee_ids = _resolve_assignee_ids(parsed, ctx)
    if not config.has_permission(ctx, "view_others"):
        if parsed.get("assignee_condition") in {"other", "unassigned", "assigned"}:
            raise PermissionError("Your role can only view progress for tasks assigned to you.")
        if assignee_ids and any(user_id != ctx.user_id for user_id in assignee_ids):
            raise PermissionError("Your role can only view progress for tasks assigned to you.")
        assignee_ids = [ctx.user_id]
    relevant = _filter_items(items, parsed, schema, assignee_ids, default_pending=False)

    has_completed = bool(slack_tools.column(
        schema, keys=slack_tools.COMPLETED_KEYS, names={"Completed"},
        types={"todo_completed", "completed", "checkbox"}) or slack_tools.column(
            schema, keys=slack_tools.STATUS_KEYS, names={"Status", "State"},
            types={"select", "multi_select"}))
    available_fields = set()
    if has_completed and (config.can_read_field(ctx, "completed") or config.can_read_field(ctx, "status")):
        available_fields.add("status")
    if slack_tools.column(schema, keys=slack_tools.DUE_KEYS, names={"Due Date", "Date"}) and config.can_read_field(ctx, "due_date"):
        available_fields.add("due_date")
    if slack_tools.column(schema, keys=slack_tools.PRIORITY_KEYS, names={"Priority"}) and config.can_read_field(ctx, "priority"):
        available_fields.add("priority")
    if slack_tools.column(schema, keys=slack_tools.ASSIGNEE_KEYS, names={"Assignee", "Owner"}) and config.can_read_field(ctx, "assignee"):
        available_fields.add("assignee")

    metrics = parsed.get("analytics_metrics") or ["overview"]
    member_names = {}
    if set(metrics) & {"workload", "summary"}:
        member_names = {member.get("id"): member.get("name")
                        for member in slack_tools.list_workspace_members() if member.get("id")}
    def member_name(user_id):
        return member_names.get(user_id) or user_name(user_id)

    report = progress_engine.calculate_progress(
        relevant, schema, today=current_date(), name_for_user=member_name,
        available_fields=available_fields,
        metrics=metrics,
        period=parsed.get("analytics_period"),
        comparison_periods=parsed.get("analytics_comparison"),
        requested_statuses=parsed.get("statuses"))
    response = progress_engine.render_progress(
        report, lambda records, title: format_items(records, schema, title, ctx))
    if report.displayed_items:
        unique = {slack_tools.extract_item_id(item): item for item in report.displayed_items}
        store_view(memory_key, list(unique.values()), schema=schema, ctx=ctx,
                   query_filter={**deepcopy(parsed), "resolved_assignee_ids": assignee_ids})
    return response


def _authorized_read_items(parsed, ctx, *, default_pending=False):
    """Fetch one current List snapshot and apply the existing RBAC/filter path."""
    if not config.has_permission(ctx, "view"):
        raise PermissionError("Your role cannot view action items.")
    if not ctx.list_id:
        raise ValueError("This channel is not mapped to a Slack List.")
    schema = slack_tools.get_list_schema(ctx.list_id)
    items = slack_tools.list_action_items(ctx, ctx.list_id)
    if (parsed.get("assignee") or parsed.get("assignees") or parsed.get("assignee_self")
            or parsed.get("assignee_condition")) and not config.can_read_field(ctx, "assignee"):
        raise PermissionError("Your role cannot use assignee data for this analysis.")
    if parsed.get("priority") and not config.can_read_field(ctx, "priority"):
        raise PermissionError("Your role cannot use priority data for this analysis.")
    if (parsed.get("overdue") or parsed.get("due_today") or parsed.get("due_this_week")
            or (parsed.get("temporal_filter") or {}).get("field") == "due_date") and not config.can_read_field(ctx, "due_date"):
        raise PermissionError("Your role cannot use due-date data for this analysis.")
    assignee_ids = _resolve_assignee_ids(parsed, ctx)
    if not config.has_permission(ctx, "view_others"):
        if parsed.get("assignee_condition") in {"other", "unassigned", "assigned"}:
            raise PermissionError("Your role can only view tasks assigned to you.")
        if assignee_ids and any(user_id != ctx.user_id for user_id in assignee_ids):
            raise PermissionError("Your role can only view tasks assigned to you.")
        assignee_ids = [ctx.user_id]
    return schema, items, _filter_items(items, parsed, schema, assignee_ids, default_pending), assignee_ids


def weekly_period(today=None):
    """Return the local Monday-Sunday reporting period containing ``today``."""
    today = today or current_date()
    start = today - timedelta(days=today.weekday())
    return start, start + timedelta(days=6)


def handle_weekly_summary(parsed, ctx, memory_key, *, period_start=None, period_end=None):
    """Render a read-only weekly report from authorized List state and audit facts."""
    schema, _, visible, assignee_ids = _authorized_read_items(
        parsed, ctx, default_pending=False)
    if not (config.can_read_field(ctx, "status") or config.can_read_field(ctx, "completed")):
        raise PermissionError("Your role cannot read task status for a weekly summary.")
    period_start, default_end = weekly_period(period_start or current_date())
    period_end = period_end or default_end
    timezone = ZoneInfo(os.getenv("DEADLINE_REMINDER_TIMEZONE", "Asia/Kathmandu"))
    since = datetime.combine(period_start, datetime.min.time(), timezone).timestamp()
    until = datetime.combine(period_end + timedelta(days=1), datetime.min.time(), timezone).timestamp()
    visible_by_id = {slack_tools.extract_item_id(item): item for item in visible}
    events = audit_log.history(
        DB_PATH, ctx.list_id, tuple(visible_by_id), limit=10000, since=since, until=until)

    created_ids = {entry["item_id"] for entry in events if entry["operation"] == "create"}
    completed_ids = {
        entry["item_id"] for entry in events
        if entry["operation"] == "complete" or (
            (entry.get("after") or {}).get("completed") is True
            and (entry.get("before") or {}).get("completed") is not True)
    }
    pending = [item for item in visible if not slack_tools.extract_completed(item, schema)
               and str(slack_tools.extract_status(item, schema) or "").casefold()
               not in {"cancelled", "canceled", "closed"}]
    today = current_date()
    overdue = []
    if config.can_read_field(ctx, "due_date"):
        for item in pending:
            try:
                due = date.fromisoformat(str(slack_tools.extract_due_date(item, schema))[:10])
            except (TypeError, ValueError):
                continue
            if due < today:
                overdue.append(item)

    priorities = {"P1": 0, "P2": 0, "P3": 0, "P4": 0}
    if config.can_read_field(ctx, "priority"):
        for item in pending:
            priority = slack_tools.extract_priority(item, schema)
            if priority in priorities:
                priorities[priority] += 1

    created = [visible_by_id[item_id] for item_id in created_ids if item_id in visible_by_id]
    completed = [visible_by_id[item_id] for item_id in completed_ids if item_id in visible_by_id]
    period_label = (f"{slack_presentation.compact_date(period_start.isoformat(), today)}–"
                    f"{slack_presentation.compact_date(period_end.isoformat(), today)}")
    pending_rows = _task_rows(pending, schema, ctx)
    pending_table = (slack_presentation.render_section(
        f"Pending · {len(pending_rows)}",
        slack_presentation.render_slack_table(
            ("Task", "Owner", "Priority", "Due"),
            [(row.name, row.assignee or "Unassigned", row.priority or "—",
              slack_presentation.due_label(row, today) if row.show_due else "—")
             for row in pending_rows])) if pending_rows else
        slack_presentation.render_section(
            "Pending · 0", "No pending action items this week."))
    lines = ["📊 *Weekly Action Items Summary*", period_label, "", pending_table, "",
             f"*Created:* {len(created)} · *Completed:* {len(completed)} · "
             f"*Overdue:* {len(overdue)}"]
    if config.can_read_field(ctx, "priority"):
        lines.append("*Priority:* " + " · ".join(
            f"{priority} {priorities[priority]}" for priority in ("P1", "P2", "P3")))

    analysis_schema = _readable_analysis_schema(schema, ctx)
    snapshot = project_intelligence.normalize_task_snapshot(pending, analysis_schema)
    insights = project_intelligence.generate_weekly_insights(
        snapshot, analysis_schema, user_name, today)
    if insights:
        lines.extend(("", "*Insights*", *(f"• {slack_presentation.text(value)}" for value in insights)))
    logger.info(
        "weekly_insights_generated list_id=%s actor_id=%s pending_count=%d insight_count=%d "
        "llm_used=false llm_call_count=0",
        ctx.list_id, ctx.user_id, len(pending), len(insights))

    def compact_section(title, items, *, include_owner=False, include_priority=False):
        if not items:
            return
        lines.extend(("", f"*{title}*"))
        for item in items[:5]:
            values = [slack_presentation.text(slack_tools.extract_item_name(item, schema))]
            if include_owner and config.can_read_field(ctx, "assignee"):
                values.append(slack_presentation.text(item_assignees(item, schema) or "Unassigned"))
            if include_priority and config.can_read_field(ctx, "priority"):
                values.append(slack_presentation.text(
                    slack_tools.extract_priority(item, schema) or "No priority"))
            lines.append("• " + " — ".join(values))
        if len(items) > 5:
            lines.append(f"• …and {len(items) - 5} more")

    compact_section("Completed", completed)
    compact_section("Overdue", overdue, include_owner=True, include_priority=True)
    important = list(dict.fromkeys(
        slack_tools.extract_item_id(item) for item in [*pending, *completed]))
    if memory_key and important:
        store_view(memory_key, [visible_by_id[item_id] for item_id in important], schema, ctx,
                   query_filter={**deepcopy(parsed), "resolved_assignee_ids": assignee_ids})
    logger.info(
        "weekly_summary_generated list_id=%s actor_id=%s period_start=%s period_end=%s "
        "created=%d completed=%d pending=%d overdue=%d",
        ctx.list_id, ctx.user_id, period_start.isoformat(), period_end.isoformat(),
        len(created), len(completed), len(pending), len(overdue))
    return "\n".join(lines)


def _readable_analysis_schema(schema, ctx):
    """Hide unreadable columns from deterministic insight calculations."""
    if not isinstance(schema, dict) or not isinstance(schema.get("schema"), list):
        return schema
    readable = []
    for field in schema["schema"]:
        raw_identities = [str(field.get(attribute) or "").strip().casefold()
                          for attribute in ("key", "name", "title")]
        identities = set(raw_identities)
        if identities & slack_tools.NAME_KEYS:
            canonical = "name"
        elif identities & slack_tools.ASSIGNEE_KEYS:
            canonical = "assignee"
        elif identities & slack_tools.DUE_KEYS:
            canonical = "due_date"
        elif identities & slack_tools.PRIORITY_KEYS:
            canonical = "priority"
        elif identities & (slack_tools.COMPLETED_KEYS | slack_tools.STATUS_KEYS):
            canonical = "status"
        else:
            canonical = next((value for value in raw_identities if value), "")
        if config.can_read_field(ctx, canonical) or (
                canonical == "status" and config.can_read_field(ctx, "completed")):
            readable.append(field)
    value = deepcopy(schema)
    value["schema"] = readable
    return value


def _health_response(records, schema, ctx, title="Task Health"):
    if not records:
        return f"*{title}*\n\nNo matching tasks found."
    blocks = []
    for record in records:
        name = slack_tools.extract_item_name(record.item, schema)
        details = [record.level, *record.reasons]
        if config.can_read_field(ctx, "assignee"):
            details.append(item_assignees(record.item, schema) or "Unassigned")
        blocks.append(f"{record.icon} *{slack_presentation.text(name)}* — " + " · ".join(
            slack_presentation.text(value) for value in details))
    return f"*{title}* · {len(blocks)} task{'s' if len(blocks) != 1 else ''}\n" + "\n".join(blocks)


def handle_focus(parsed, ctx, memory_key):
    """Render an explainable, read-only ordering of the requester's work."""
    focus_query = deepcopy(parsed)
    focus_query.pop("due_today", None)
    focus_query.pop("temporal_filter", None)
    schema, _, relevant, assignee_ids = _authorized_read_items(
        focus_query, ctx, default_pending=True)
    logger.info(
        "task_intelligence_started actor_id=%s list_id=%s task_count=%d operation=focus",
        ctx.user_id, ctx.list_id, len(relevant))
    analysis_schema = _readable_analysis_schema(schema, ctx)
    snapshot = project_intelligence.normalize_task_snapshot(relevant, analysis_schema)
    entries = project_intelligence.calculate_daily_focus(snapshot, analysis_schema, current_date())
    logger.info(
        "task_focus_calculated actor_id=%s list_id=%s task_count=%d pending_count=%d "
        "llm_used=false llm_call_count=0",
        ctx.user_id, ctx.list_id, len(relevant), len(entries))
    if not entries:
        state = _state(ctx)
        state.update(last_intent="focus", last_task_id=None, last_task_name=None,
                     displayed_tasks=[])
        _save_state(ctx, state)
        return slack_presentation.join_sections(
            "*🎯 Focus Today*", "No pending action items are assigned to you.")
    displayed = entries[:5]
    if memory_key:
        store_view(memory_key, [entry.item for entry in displayed], schema, ctx,
                   query_filter={**deepcopy(focus_query), "resolved_assignee_ids": assignee_ids})
        # Focus ordering supplies a primary referent only when its top factual
        # rank differs from the runner-up. Equal-ranked tasks remain ambiguous.
        primary = displayed[0] if displayed else None
        runner_up = displayed[1] if len(displayed) > 1 else None
        rank = lambda entry: (entry.category, entry.due_date, entry.priority, entry.reason)
        state = _state(ctx)
        state.update(last_intent="focus",
                     last_task_id=(primary.item_id if primary and (
                         not runner_up or rank(primary) != rank(runner_up)) else None),
                     last_task_name=(slack_tools.extract_item_name(primary.item, schema)
                                     if primary and (not runner_up or rank(primary) != rank(runner_up))
                                     else None))
        _save_state(ctx, state)
    today = current_date()
    immediate_count = sum(entry.category == "immediate" for entry in entries)
    sections = ["*🎯 Focus Today*"]
    if immediate_count:
        sections.append(
            f"*{immediate_count} task{'s' if immediate_count != 1 else ''} need"
            f"{'s' if immediate_count == 1 else ''} attention*")
    if not any(entry.due_date == today for entry in entries):
        sections.extend(("No action items are due today.",
                         f"*Pending work:* {len(entries)} task{'s' if len(entries) != 1 else ''}"))
    for category, heading in (("immediate", "🔴 Immediate Attention"),
                              ("upcoming", "📅 Upcoming")):
        grouped = [entry for entry in displayed if entry.category == category]
        if not grouped:
            continue
        table_rows = []
        for entry in grouped:
            name = slack_presentation.text(slack_tools.extract_item_name(entry.item, schema))
            if entry.due_date and entry.due_date < today:
                days = (today - entry.due_date).days
                due = f"🔴 {days}d overdue"
            elif entry.due_date == today:
                due = "🟡 Due today"
            elif entry.due_date == today + timedelta(days=1):
                due = "Due tomorrow"
            else:
                due = (slack_presentation.compact_date(entry.due_date.isoformat(), today)
                       if entry.due_date else "No due date")
            table_rows.append((name, entry.priority or "—", due))
        sections.append(slack_presentation.render_section(
            heading, slack_presentation.render_slack_table(
                ("Task", "Priority", "Status" if category == "immediate" else "Due"),
                table_rows)))
    summary = [f"{immediate_count} task{'s' if immediate_count != 1 else ''} require"
               f"{'s' if immediate_count == 1 else ''} immediate attention."]
    if len(entries) > len(displayed):
        summary.append(f"Showing {len(displayed)} focus items from {len(entries)} pending tasks.")
    sections.append(slack_presentation.render_section("Summary", *summary))
    return slack_presentation.join_sections(*sections)


def handle_weekly_focus(parsed, ctx, memory_key):
    """Render the requester's pending carryover and deadlines for this week."""
    focus_query = deepcopy(parsed)
    # Apply the weekly horizon after authorization so overdue carryover is not
    # discarded by the generic due-this-week list filter.
    focus_query.pop("due_this_week", None)
    focus_query.pop("temporal_filter", None)
    schema, _, relevant, assignee_ids = _authorized_read_items(
        focus_query, ctx, default_pending=True)
    analysis_schema = _readable_analysis_schema(schema, ctx)
    snapshot = project_intelligence.normalize_task_snapshot(relevant, analysis_schema)
    today = current_date()
    week_start, week_end = weekly_period(today)
    weekly_snapshot = [
        task for task in snapshot
        if not task.completed and task.due_date and task.due_date <= week_end
    ]
    entries = project_intelligence.calculate_daily_focus(
        weekly_snapshot, analysis_schema, today)
    logger.info(
        "task_weekly_focus_calculated actor_id=%s list_id=%s task_count=%d "
        "week_start=%s week_end=%s llm_used=false llm_call_count=0",
        ctx.user_id, ctx.list_id, len(entries), week_start, week_end)
    if not entries:
        state = _state(ctx)
        state.update(last_intent="weekly_focus", last_task_id=None,
                     last_task_name=None, displayed_tasks=[])
        _save_state(ctx, state)
        return slack_presentation.join_sections(
            "*🎯 Weekly Focus*",
            f"*{week_start.strftime('%b %-d')}–{week_end.strftime('%b %-d')}*",
            "No pending action items require attention this week.")
    displayed = entries[:10]
    if memory_key:
        store_view(memory_key, [entry.item for entry in displayed], schema, ctx,
                   query_filter={**deepcopy(focus_query),
                                 "resolved_assignee_ids": assignee_ids})
    sections = ["*🎯 Weekly Focus*",
                f"*{week_start.strftime('%b %-d')}–{week_end.strftime('%b %-d')}*"]
    for heading, predicate in (
            ("🔴 Overdue Carryover", lambda entry: entry.due_date < today),
            ("📅 Due This Week", lambda entry: today <= entry.due_date <= week_end)):
        grouped = [entry for entry in displayed if predicate(entry)]
        if not grouped:
            continue
        table_rows = []
        for entry in grouped:
            name = slack_presentation.text(slack_tools.extract_item_name(entry.item, schema))
            owner = item_assignees(entry.item, schema) or "Unassigned"
            due = (("🔴 " if entry.due_date < today else "")
                   + slack_presentation.compact_date(entry.due_date.isoformat(), today))
            table_rows.append((name, entry.priority or "—", owner, due))
        sections.append(slack_presentation.render_section(
            heading, slack_presentation.render_slack_table(
                ("Task", "Priority", "Owner", "Due"), table_rows)))
    summary = [f"{len(entries)} pending task{'s' if len(entries) != 1 else ''} "
               f"require{'s' if len(entries) == 1 else ''} attention this week."]
    if len(entries) > len(displayed):
        summary.append(f"Showing {len(displayed)} focus items from {len(entries)} pending tasks.")
    sections.append(slack_presentation.render_section("Summary", *summary))
    return slack_presentation.join_sections(*sections)


def _risk_response(records, schema, ctx):
    if not records:
        return "⚠️ *Task Risk*\n\nNo risky pending tasks were identified from the available task data."
    lines = [f"⚠️ *Task Risk* · {len(records)}"]
    for record in records:
        item = record.item
        name = slack_presentation.text(slack_tools.extract_item_name(item, schema))
        details = []
        if config.can_read_field(ctx, "priority"):
            details.append(slack_tools.extract_priority(item, schema) or "No priority")
        if config.can_read_field(ctx, "assignee"):
            details.append(item_assignees(item, schema) or "Unassigned")
        if config.can_read_field(ctx, "due_date"):
            details.append(slack_presentation.compact_date(
                slack_tools.extract_due_date(item, schema), current_date()))
        lines.append(f"• *{name}* — {' · '.join(slack_presentation.text(value) for value in details)}")
        lines.append("  " + " · ".join(slack_presentation.text(reason) for reason in record.reasons))
    return "\n".join(lines)


def _health_summary_response(summary, schema, ctx, workload=None):
    lines = ["🩺 *Action Items Health*", "", "*Overview*",
             f"• Pending: {summary.pending}", f"• Completed: {summary.completed}",
             f"• Overdue: {summary.overdue}"]
    if config.can_read_field(ctx, "priority"):
        lines.append(f"• P1: {summary.p1}")
    if config.can_read_field(ctx, "assignee"):
        lines.append(f"• Unassigned: {summary.unassigned}")
    if config.can_read_field(ctx, "due_date"):
        lines.append(f"• Due <48h: {summary.due_within_48h}")
    attention = []
    if summary.overdue:
        attention.append(f"{summary.overdue} overdue task{'s' if summary.overdue != 1 else ''}")
    if summary.due_within_48h:
        attention.append(f"{summary.due_within_48h} task{'s' if summary.due_within_48h != 1 else ''} due within 48 hours")
    if summary.unassigned:
        attention.append(
            f"{summary.unassigned} task{'s are' if summary.unassigned != 1 else ' is'} unassigned")
    if attention:
        lines.extend(("", "⚠️ *Attention*", *(f"• {value}" for value in attention)))
    if summary.risks:
        top = summary.risks[0].item
        details = [slack_tools.extract_priority(top, schema) or "No priority"]
        if config.can_read_field(ctx, "assignee"):
            details.append(item_assignees(top, schema) or "Unassigned")
        if config.can_read_field(ctx, "due_date"):
            details.append(slack_presentation.compact_date(
                slack_tools.extract_due_date(top, schema), current_date()))
        lines.extend(("", "📌 *Highest Attention*",
                      slack_presentation.text(slack_tools.extract_item_name(top, schema)),
                      " · ".join(slack_presentation.text(value) for value in details)))
    if workload and workload.rows:
        lines.extend(("", "👥 *Workload*"))
        for _, row in sorted(
                ((user_id, row) for user_id, row in workload.rows.items() if row["pending"]),
                key=lambda pair: (-pair[1]["pending"], pair[1]["name"].casefold())):
            lines.append(f"• {slack_presentation.text(row['name'])} — {row['pending']} pending")
    return "\n".join(lines)


def handle_health(parsed, ctx, memory_key):
    summary_mode = parsed.get("health_summary", False)
    schema, items, relevant, assignee_ids = _authorized_read_items(
        parsed, ctx, default_pending=not summary_mode)
    if reference_from(parsed) or parsed.get("task_name"):
        relevant = list(resolve_target_set(parsed, items, schema, memory_key, ctx, "inspect").items)
        if not config.has_permission(ctx, "view_others") and any(
                ctx.user_id not in slack_tools.extract_assignee_ids(item, schema) for item in relevant):
            raise PermissionError("Your role can only view tasks assigned to you.")
    status_field = slack_tools.column(
        schema, keys=slack_tools.COMPLETED_KEYS | slack_tools.STATUS_KEYS,
        names={"Completed", "Status", "State"})
    due_field = slack_tools.column(schema, keys=slack_tools.DUE_KEYS, names={"Due Date", "Date"})
    if status_field and not (config.can_read_field(ctx, "status") or config.can_read_field(ctx, "completed")):
        raise PermissionError("Your role cannot read task status for health analysis.")
    if due_field and not config.can_read_field(ctx, "due_date"):
        raise PermissionError("Your role cannot read due dates for health analysis.")
    analysis_schema = _readable_analysis_schema(schema, ctx)
    logger.info(
        "task_intelligence_started actor_id=%s list_id=%s task_count=%d llm_used=false llm_call_count=0",
        ctx.user_id, ctx.list_id, len(relevant))
    snapshot = project_intelligence.normalize_task_snapshot(relevant, analysis_schema)
    if summary_mode:
        summary = project_intelligence.generate_task_health(snapshot, analysis_schema, current_date())
        workload = None
        if config.can_read_field(ctx, "assignee"):
            members = slack_tools.list_workspace_members()
            if not config.has_permission(ctx, "view_others"):
                members = [member for member in members if member["id"] == ctx.user_id]
            names = {member["id"]: member["name"] for member in members}
            workload = project_intelligence.calculate_workload(
                snapshot, analysis_schema,
                lambda user_id: names.get(user_id) or user_name(user_id),
                current_date(), names)
        logger.info(
            "task_health_generated actor_id=%s list_id=%s task_count=%d pending_count=%d risk_count=%d",
            ctx.user_id, ctx.list_id, len(relevant), summary.pending, len(summary.risks))
        return _health_summary_response(summary, schema, ctx, workload)
    if parsed.get("risk_view"):
        records = project_intelligence.analyze_task_risks(snapshot, analysis_schema, current_date())
        if records:
            store_view(memory_key, [record.item for record in records], schema, ctx,
                       query_filter={**deepcopy(parsed), "resolved_assignee_ids": assignee_ids})
        logger.info(
            "task_risk_analysis_completed actor_id=%s list_id=%s task_count=%d risk_count=%d",
            ctx.user_id, ctx.list_id, len(relevant), len(records))
        return _risk_response(records, schema, ctx)
    records = project_intelligence.calculate_health(
        relevant, analysis_schema, current_date(), attention_only=parsed.get("attention_only", False))
    if records:
        store_view(memory_key, [record.item for record in records], schema, ctx,
                   query_filter={**deepcopy(parsed), "resolved_assignee_ids": assignee_ids})
    title = "Tasks Needing Attention" if parsed.get("attention_only") else "Task Health"
    return _health_response(records, schema, ctx, title)


def handle_intelligence_summary(parsed, ctx, memory_key):
    """Render predictive facts from one authorized normalized task snapshot."""
    schema, _, visible, _ = _authorized_read_items(
        {"intent": "list"}, ctx, default_pending=False)
    analysis_schema = _readable_analysis_schema(schema, ctx)
    snapshot = project_intelligence.normalize_task_snapshot(visible, analysis_schema)
    today = current_date()
    summary = predictive_intelligence.build_predictive_summary(snapshot, today)
    mode = parsed.get("intelligence_mode") or "summary"
    logger.info(
        "task_predictive_intelligence_generated actor_id=%s list_id=%s mode=%s "
        "task_count=%d pending_count=%d emerging_risk_count=%d cluster_count=%d "
        "snapshot_reused=true llm_used=false llm_call_count=0",
        ctx.user_id, ctx.list_id, mode, len(snapshot), summary.pending,
        len(summary.emerging_risks), len(summary.deadline_clusters))

    def owner_names(owner_ids):
        if not config.can_read_field(ctx, "assignee"):
            return None
        return ", ".join(user_name(value) for value in owner_ids) or "Unassigned"

    def risk_cards():
        cards = []
        for risk in summary.emerging_risks[:5]:
            metadata = []
            if config.can_read_field(ctx, "priority"):
                metadata.append(risk.priority or "No priority")
            owners = owner_names(risk.owner_ids)
            if owners:
                metadata.append(owners)
            if config.can_read_field(ctx, "due_date") and risk.due_date:
                metadata.append(slack_presentation.compact_date(risk.due_date.isoformat(), today))
            cards.append(slack_presentation.render_task_card(
                risk.task_name, metadata,
                detail=" · ".join(risk.evidence)))
        return cards

    if mode == "emerging_risks":
        sections = ["*🧠 Predictive Task Intelligence*"]
        risks = summary.emerging_risks[:5]
        if risks:
            rows = []
            reasons = []
            for risk in risks:
                rows.append((
                    risk.task_name,
                    risk.priority or "—",
                    owner_names(risk.owner_ids) or "Restricted",
                    "High" if risk.level == "high" else "Elevated"))
                reasons.append(
                    f"• *{slack_presentation.text(risk.task_name)}* — "
                    + " · ".join(slack_presentation.text(value) for value in risk.evidence))
            sections.append(slack_presentation.render_section(
                "⚠️ Emerging Risks", slack_presentation.render_slack_table(
                    ("Task", "Priority", "Owner", "Pressure"), rows)))
            sections.append(slack_presentation.render_section("Why", "\n".join(reasons)))
            sections.append(slack_presentation.render_section(
                "Recommended Next Step",
                "Review the highlighted deadlines and ownership before pressure increases."))
        else:
            sections.append("No emerging risks were identified from the currently known task data.")
        sections.append("_No changes were made._")
        return slack_presentation.join_sections(*sections)

    if mode == "deadline_pressure":
        sections = ["*📅 Deadline Pressure*",
                    f"{summary.due_within_48h} due within 48 hours · "
                    f"{summary.due_within_7d} due within 7 days"]
        if summary.deadline_clusters:
            cluster_lines = []
            for cluster in summary.deadline_clusters:
                priorities = " · ".join(
                    f"{priority} {count}" for priority, count in sorted(cluster.priority_counts.items()))
                cluster_lines.append(
                    f"• *{slack_presentation.compact_date(cluster.due_date.isoformat(), today)}* — "
                    f"{len(cluster.task_ids)} tasks" + (f" · {priorities}" if priorities else ""))
            sections.append(slack_presentation.render_section(
                "Deadline Concentration", "\n".join(cluster_lines)))
        else:
            sections.append("No near-term deadline clusters were identified.")
        sections.append("_Projection uses only currently known pending tasks._")
        return slack_presentation.join_sections(*sections)

    if mode == "workload_outlook":
        if not summary.workload:
            return slack_presentation.join_sections(
                "*📈 Workload Outlook*", "No pending workload is available.",
                "_Projection uses only currently known pending tasks._")
        workload_rows = [
            (user_name(row.owner_id) if row.owner_id else "Unassigned",
             row.pending, row.p1, row.overdue)
            for row in summary.workload]
        pressure = [f"{summary.due_within_48h} tasks due within 48 hours"]
        if summary.deadline_clusters:
            cluster = summary.deadline_clusters[0]
            pressure.append(
                f"{len(cluster.task_ids)} tasks converge on "
                f"{slack_presentation.compact_date(cluster.due_date.isoformat(), today)}")
        sections = ["*📈 Workload Outlook*",
                    slack_presentation.render_slack_table(
                        ("Owner", "Pending", "P1", "Overdue"), workload_rows),
                    slack_presentation.render_section(
                        "📅 Deadline Pressure", "\n".join(pressure))]
        if summary.emerging_risks:
            sections.append(slack_presentation.render_section(
                "⚠️ Emerging Risk",
                "Workload and deadlines create measurable near-term pressure."))
        sections.append("_Projection uses only currently known pending tasks._")
        return slack_presentation.join_sections(*sections)

    sections = ["*🧠 Task Intelligence*",
                slack_presentation.render_section(
                    "Team",
                    f"{summary.pending} pending · {summary.completed} completed · "
                    f"{summary.overdue} overdue")]
    if config.can_read_field(ctx, "priority"):
        counts = summary.priority_counts
        sections.append(slack_presentation.render_section(
            "Priority", f"{counts['P1']} P1 · {counts['P2']} P2 · {counts['P3']} P3"))
    if config.can_read_field(ctx, "due_date"):
        sections.append(slack_presentation.render_section(
            "📅 Deadline Pressure",
            f"{summary.due_within_48h} tasks due within 48 hours · "
            f"{summary.due_within_7d} due within 7 days"))
    if config.can_read_field(ctx, "assignee"):
        sections.append(slack_presentation.render_section(
            "👥 Ownership", f"{summary.unassigned} unassigned pending tasks"))
        active = [row for row in summary.workload if row.owner_id]
        if active:
            largest = active[0]
            sections.append(slack_presentation.render_section(
                "⚖️ Workload",
                f"{slack_presentation.text(user_name(largest.owner_id))} has the largest visible "
                f"pending workload ({largest.pending} tasks)."))
    cards = risk_cards()
    if cards:
        sections.append(slack_presentation.render_section(
            "⚠️ Emerging Risk", slack_presentation.render_item_list(cards)))
    if summary.overdue:
        recommendation = "Review overdue work, starting with P1 tasks."
    elif summary.due_within_48h:
        recommendation = "Review tasks due within 48 hours."
    elif summary.unassigned:
        recommendation = "Review unassigned pending work."
    elif summary.pending:
        recommendation = "Review the next pending deadline."
    else:
        recommendation = "No pending action is required."
    sections.extend((slack_presentation.render_section("🎯 Recommended Focus", recommendation),
                     "_No actions were taken._"))
    return slack_presentation.join_sections(*sections)


def _command_center_autopilot(alerts, ctx, current_risks):
    current_keys = {(f"risk:{risk.risk_type}", risk.task_id) for risk in current_risks}
    recommendations = [smart_task_autopilot.prepare_recommendation(
        alert_id=alert.alert_id, payload=alert.payload, requesting_user=ctx.user_id,
        owner_name=(", ".join(user_name(value) for value in alert.payload.get("owner_ids") or [])
                    or None),
        today=current_date(), created_at=alert.created_at, status=alert.status)
        for alert in alerts if (alert.event_type, alert.task_id) in current_keys]
    selected = next((value for value in recommendations if value.executable),
                    recommendations[0] if recommendations else None)
    if not selected:
        return [], None
    return recommendations, selected


def handle_command_center(parsed, ctx, memory_key):
    """Orchestrate existing intelligence over one authorized normalized snapshot."""
    mode = parsed.get("command_center_mode") or "overview"
    if mode in {"risk_followup", "prepare_message"}:
        owner_id = _state(ctx).get("command_center_owner_id")
        if not owner_id and not parsed.get("context_task_id"):
            raise ValueError(
                "I don't have an active Command Center risk in this conversation. "
                "Ask why a named workspace member is at risk first.")
        if owner_id:
            parsed = {**parsed, "resolved_assignee_ids": [owner_id]}
    schema, _, visible, assignee_ids = _authorized_read_items(
        parsed, ctx, default_pending=False)
    analysis_schema = _readable_analysis_schema(schema, ctx)
    snapshot = project_intelligence.normalize_task_snapshot(visible, analysis_schema)
    logger.info("command_center intent=%s actor_id=%s list_id=%s llm_used=false llm_call_count=0",
                mode, ctx.user_id, ctx.list_id)
    logger.info("command_center_snapshot_reused list_id=%s task_count=%d", ctx.list_id, len(snapshot))

    engine = _sentinel_engine()
    evaluation = engine.evaluate(
        ctx.list_id, snapshot,
        datetime.now(ZoneInfo(os.getenv("DEADLINE_REMINDER_TIMEZONE", "Asia/Kathmandu"))),
        persist_snapshot=False)
    by_id = {task.item_id: task.item for task in snapshot}
    alerts = _visible_sentinel_alerts(engine, ctx.list_id, by_id)
    recommendations, selected = _command_center_autopilot(alerts, ctx, evaluation.risks)

    if mode in {"owner_risk", "risk_followup", "prepare_message"}:
        context_task_id = parsed.get("context_task_id")
        context_task = next((task for task in snapshot
                             if task.item_id == context_task_id), None)
        if context_task_id and not context_task:
            raise ValueError("That recent task no longer exists or is not authorized for you.")
        owner_ids = (list(context_task.owner_ids) if context_task
                     else parsed.get("resolved_assignee_ids") or assignee_ids)
        if len(owner_ids) != 1:
            raise ValueError("That task needs one clear owner before a reminder can be prepared.")
        owner_id = owner_ids[0]
        owned, risks = command_center.owner_risks(snapshot, owner_id, today=current_date())
        if context_task:
            owned = [task for task in owned if task.item_id == context_task.item_id]
            risks = [risk for risk in risks if context_task.item_id in risk.task_ids]
        owner = slack_presentation.text(user_name(owner_id))
        if not owned:
            return f"*{owner} — Risk Explanation*\n\nNo authorized pending tasks were found for this member."
        if not risks:
            return f"*{owner} — Risk Explanation*\n\nNo active deterministic risks were identified."
        lines = [f"*{owner} — Risk Explanation*"]
        task_risks = [risk for risk in risks if risk.risk_type != "combined_workload_risk"]
        for risk in task_risks[:5]:
            task = next((value for value in snapshot if value.item_id == risk.task_id), None)
            if not task:
                continue
            due = slack_presentation.compact_date(
                task.due_date.isoformat(), current_date()) if task.due_date else "No due date"
            lines.extend(("", f"• *{slack_presentation.text(task.name)}* — {task.priority or 'No priority'} · {due}",
                          "  " + " · ".join(slack_presentation.text(value) for value in risk.reasons)))
        workload = next((risk for risk in risks if risk.risk_type == "combined_workload_risk"), None)
        if workload:
            lines.extend(("", f"{len(workload.task_ids)} P1 tasks have deadlines within "
                          f"{engine.settings.warning_days * 24} hours."))
        lines.extend(("", "*Recommended Action*",
                      f"Review {owner}'s urgent deadlines and confirm ownership/status.", "",
                      "*No action has been taken automatically.*"))
        if mode == "prepare_message":
            owned_ids = {task.item_id for task in owned}
            prepared = next((value for value in recommendations
                             if value.executable and next(
                                 (alert.task_id for alert in alerts
                                  if alert.alert_id == value.alert_id), None) in owned_ids), None)
            if prepared and prepared.prepared_message:
                alert = next(value for value in alerts if value.alert_id == prepared.alert_id)
                engine.store.save_display(
                    ctx.list_id, ctx.channel_id, ctx.user_id, [alert.alert_id], time.time())
                lines[-1:-1] = ["*Prepared Message*",
                                f"> {slack_presentation.text(prepared.prepared_message)}", "",
                                "*Approval*", "`send sentinel alert 1`", ""]
            else:
                lines[-1:-1] = ["No executable reminder is available for this risk.", ""]
        state = _state(ctx)
        state.update(command_center_owner_id=owner_id,
                     command_center_checked_at=time.time(), last_intent="owner_risk",
                     last_task_id=None, last_task_name=None, displayed_tasks=[])
        _save_state(ctx, state)
        logger.info("command_center_sections_generated mode=%s risk_count=%d", mode, len(risks))
        return "\n".join(lines)

    report = command_center.build_report(snapshot, today=current_date(), name_for_user=user_name)
    lines = ["*Smart Task Command Center*", "", "📊 *Team Status*",
             f"• {report.pending} pending · {report.completed} completed",
             f"• {report.priorities['P1']} P1 · {report.priorities['P2']} P2 · {report.priorities['P3']} P3 among pending tasks",
             f"• {report.overdue} overdue · {report.due_this_week} due this week"]
    if report.critical:
        lines.extend(("", "🔴 *Critical*"))
        for task in report.critical:
            status = "overdue" if task.due_date and task.due_date < current_date() else "due today"
            owner = ", ".join(user_name(value) for value in task.owner_ids) or "Unassigned"
            lines.append(f"• *{slack_presentation.text(task.name)}* — {status} — {slack_presentation.text(owner)}")
    attention = [risk for risk in report.risks
                 if risk.risk_type in {"combined_workload_risk", "unassigned_deadline_risk"}]
    if attention:
        lines.extend(("", "🟠 *Needs Attention*"))
        for risk in attention[:5]:
            owner = ", ".join(user_name(value) for value in risk.owner_ids) or "Unassigned"
            lines.append(f"• {slack_presentation.text(owner)} — " +
                         " · ".join(slack_presentation.text(value) for value in risk.reasons))
    if config.can_read_field(ctx, "due_date"):
        lines.extend(("", "📅 *Deadline Pressure*",
                      f"• {report.predictive.due_within_48h} due within 48 hours · "
                      f"{report.predictive.due_within_7d} due within 7 days"))
        if report.predictive.deadline_clusters:
            cluster = report.predictive.deadline_clusters[0]
            lines.append(
                f"• {len(cluster.task_ids)} tasks converge on "
                f"{slack_presentation.compact_date(cluster.due_date.isoformat(), current_date())}")
    if report.predictive.emerging_risks:
        lines.extend(("", "🧠 *Emerging Risk*"))
        for risk in report.predictive.emerging_risks[:3]:
            lines.append(
                f"• *{slack_presentation.text(risk.task_name)}* — "
                + " · ".join(slack_presentation.text(value) for value in risk.evidence))
    active_rows = [row for row in report.workload.rows.values() if row["pending"]]
    if active_rows:
        lines.extend(("", "👥 *Workload*"))
        for row in sorted(active_rows, key=lambda value: (-value["pending"], value["name"].casefold()))[:5]:
            lines.append(f"• {slack_presentation.text(row['name'])} — {row['pending']} pending")
    lines.extend(("", "*Autopilot*",
                  f"• {report.reminder_count} reminder{'s' if report.reminder_count != 1 else ''} prepared",
                  f"• {report.workload_review_count} workload review{'s' if report.workload_review_count != 1 else ''} recommended",
                  "• 0 actions executed automatically"))
    if selected:
        lines.extend(("", "*Prepared Recommendation*", selected.recommendation))
        if selected.prepared_message:
            lines.extend(("", "*Prepared Message*", f"> {slack_presentation.text(selected.prepared_message)}"))
        selected_alert = next(alert for alert in alerts if alert.alert_id == selected.alert_id)
        if selected.executable:
            engine.store.save_display(
                ctx.list_id, ctx.channel_id, ctx.user_id, [selected_alert.alert_id], time.time())
            lines.extend(("", "*Approval*", "`send sentinel alert 1`"))
    lines.extend(("", "💡 *System Insight*", report.insight, "",
                  "*Recommended Next Step*", report.next_step, "",
                  "*No action has been taken automatically.*"))
    state = _state(ctx)
    state.update(command_center_checked_at=time.time(), last_intent="command_center",
                 last_task_id=None, last_task_name=None, displayed_tasks=[])
    _save_state(ctx, state)
    logger.info("command_center_sections_generated mode=overview sections=%d risk_count=%d",
                6, len(report.risks))
    return "\n".join(lines)


def _visual_text(dataset):
    rows = "\n".join(f"• {slack_presentation.text(label)}: {value}"
                     for label, value in dataset.series)
    return (f"*{slack_presentation.text(dataset.title)}*\n\n{rows}\n\n"
            f"{slack_presentation.text(dataset.summary)}")


def handle_visual_analytics(parsed, ctx, memory_key):
    """Build a visual from one authorized normalized snapshot, with text fallback."""
    schema, _, visible, assignee_ids = _authorized_read_items(
        parsed, ctx, default_pending=False)
    analysis_schema = _readable_analysis_schema(schema, ctx)
    snapshot = project_intelligence.normalize_task_snapshot(visible, analysis_schema)
    kind = parsed.get("visualization_type") or "dashboard"
    requested_mode = parsed.get("response_mode") or "chart"
    requested_chart_type = parsed.get("chart_type") or "auto"
    member_names = ({member.get("id"): member.get("name")
                     for member in slack_tools.list_workspace_members() if member.get("id")}
                    if kind in {"workload", "dashboard"} else {})
    display_name = lambda user_id: member_names.get(user_id) or user_name(user_id)
    required_field = {"workload": "assignee", "priority": "priority",
                      "completion": "status", "completed_trend": "status",
                      "deadlines": "due_date"}.get(kind)
    if required_field == "status":
        readable = (config.can_read_field(ctx, "status")
                    or config.can_read_field(ctx, "completed"))
    else:
        readable = not required_field or config.can_read_field(ctx, required_field)
    if not readable:
        logger.info("visual_request intent=%s response_mode=text visualization_type=%s "
                    "records=%d fallback=true reason=field_unavailable llm_used=false",
                    kind, kind, len(snapshot))
        return f"*Visual analytics unavailable*\n\nThe {required_field.replace('_', ' ')} field is not readable for your role."

    datasets = []
    table_dataset = None
    def dataset_for(value):
        if value == "workload":
            workload = project_intelligence.calculate_workload(
                snapshot, analysis_schema, display_name, current_date(), member_names)
            return visual_analytics.workload_dataset(workload)
        return visual_analytics.build_dataset(
            snapshot, value, today=current_date(), name_for_user=display_name)

    if kind in {"all_tasks", "overdue_tasks", "upcoming_tasks"}:
        today = current_date()
        selected = [task for task in snapshot if (
            kind == "all_tasks"
            or (kind == "overdue_tasks" and not task.completed and task.due_date and task.due_date < today)
            or (kind == "upcoming_tasks" and not task.completed and task.due_date and task.due_date >= today))]
        selected.sort(key=lambda task: (task.due_date or date.max, task.name.casefold()))
        title = {"all_tasks": "Authorized Action Items", "overdue_tasks": "Overdue Action Items",
                 "upcoming_tasks": "Upcoming Deadlines"}[kind]
        rows = tuple((task.name,
                      ", ".join(display_name(value) for value in task.owner_ids) or "Unassigned",
                      task.priority or "No priority",
                      slack_presentation.compact_date(
                          task.due_date.isoformat(), today) if task.due_date else "No due date")
                     for task in selected[:25])
        table_dataset = visual_analytics.TableDataset(
            title, "authorized tasks", ("Task", "Owner", "Priority", "Due"), rows,
            f"{len(selected)} matching task{'s' if len(selected) != 1 else ''}.")
    elif kind in {"completed_trend", "created_trend"}:
        dimension = "completed" if kind == "completed_trend" else "created"
        series = progress_engine.calculate_time_series(visible, schema, dimension)
        values = tuple(series.get("values", {}).items())
        title = ("Completed Tasks Over Time" if dimension == "completed"
                 else "Created Tasks Over Time")
        summary = (f"{series.get('available', 0)} timestamped task records are available."
                   if values else "Reliable historical timestamps are unavailable; no trend was generated.")
        datasets.append(visual_analytics.VisualDataset(
            kind, title, "authorized timestamped tasks", values,
            series.get("available", 0), summary))
    elif kind == "dashboard":
        for value in ("completion", "priority", "workload", "deadlines"):
            field = {"completion": "status", "priority": "priority",
                     "workload": "assignee", "deadlines": "due_date"}[value]
            allowed = ((config.can_read_field(ctx, "status") or config.can_read_field(ctx, "completed"))
                       if field == "status" else config.can_read_field(ctx, field))
            if allowed:
                datasets.append(dataset_for(value))
    else:
        datasets.append(dataset_for(kind))
    meaningful = [dataset for dataset in datasets if dataset.meaningful]
    if table_dataset:
        fallback = (format_items([task.item for task in selected], schema, table_dataset.title, ctx)
                    if selected else f"*{table_dataset.title}*\n\nNo matching action items found.")
        if not table_dataset.meaningful:
            logger.info("visual_request intent=%s response_mode=text visualization_type=table "
                        "scope=authorized records=0 fallback=true llm_used=false llm_call_count=0", kind)
            return fallback
        try:
            logger.info("visual_generation started=true visualization_type=table")
            png = visual_analytics.render_table_png(table_dataset)
        except Exception as exc:
            logger.warning("visual_generation success=false fallback=text visualization_type=table error_type=%s",
                           type(exc).__name__)
            return "I couldn't generate the table right now, so here's the task summary instead.\n\n" + fallback
        state = _state(ctx)
        state.update(last_intent="visual_analytics", visual_context={
            "kind": kind, "response_mode": "table", "chart_type": "table", "created": time.time()})
        _save_state(ctx, state)
        logger.info("visual_request intent=%s response_mode=table visualization_type=table "
                    "scope=authorized records=%d llm_used=false llm_call_count=0", kind, len(selected))
        logger.info("visual_generation success=true visualization_type=table")
        return {"text": f"*{table_dataset.title}*\n_Scope: {table_dataset.scope}_\n{table_dataset.summary}",
                "fallback_text": fallback,
                "visual": {"content_base64": base64.b64encode(png).decode("ascii"),
                           "filename": f"task-{kind}.png", "title": table_dataset.title}}
    mode = visual_analytics.choose_response_mode(
        explicit_visual=bool(parsed.get("explicit_visual")),
        dashboard=kind == "dashboard", value_count=sum(len(value.series) for value in meaningful))
    fallback = ("*Visual Analytics*\n\nNo meaningful authorized task data is available to visualize."
                if not meaningful else "\n\n".join(_visual_text(value) for value in meaningful))
    if mode == "text" or not meaningful:
        logger.info("visual_request intent=%s response_mode=text visualization_type=%s "
                    "scope=authorized records=%d fallback=true llm_used=false llm_call_count=0",
                    kind, kind, len(snapshot))
        return fallback
    chart_type = requested_chart_type
    if chart_type == "auto":
        chart_type = ("dashboard" if kind == "dashboard" else
                      "pie" if kind in {"priority", "completion"} else
                      "line" if kind in {"completed_trend", "created_trend"} else "bar")
    if chart_type == "table":
        mode = "table"
    if chart_type == "line" and kind not in {"completed_trend", "created_trend"}:
        return ("A line chart requires real ordered or timestamped task data. "
                "I couldn't generate one without inventing a trend.\n\n" + fallback)
    try:
        logger.info("visual_generation started=true visualization_type=%s", chart_type)
        png = (visual_analytics.render_dashboard_png(meaningful) if kind == "dashboard"
               else visual_analytics.render_chart_png(meaningful[0], chart_type))
    except Exception as exc:
        logger.warning("visual_generation success=false fallback=text visualization_type=%s error_type=%s",
                       chart_type, type(exc).__name__)
        return "I couldn't generate the chart right now, so here's the summary instead.\n\n" + fallback
    state = _state(ctx)
    state.update(last_intent="visual_analytics", visual_context={
        "kind": kind, "response_mode": requested_mode,
        "chart_type": chart_type, "created": time.time()})
    _save_state(ctx, state)
    logger.info("visual_request intent=%s response_mode=%s visualization_type=%s "
                "scope=authorized records=%d generation_success=true fallback=false "
                "llm_used=false llm_call_count=0",
                kind, mode, chart_type, len(snapshot))
    title = "Smart Task Visual Dashboard" if kind == "dashboard" else meaningful[0].title
    summary = (fallback if kind == "dashboard" else
               f"*{slack_presentation.text(title)}*\n"
               f"_Scope: {slack_presentation.text(meaningful[0].scope)}_\n"
               f"{slack_presentation.text(meaningful[0].summary)}")
    logger.info("visual_generation success=true visualization_type=%s", chart_type)
    return {"text": summary, "fallback_text": fallback, "visual": {
            "content_base64": base64.b64encode(png).decode("ascii"),
            "filename": f"task-{kind}-{chart_type}.png", "title": title}}


def _orchestrator_plan(state, plan_id=None):
    plans = state.get("orchestrator_plans") or {}
    if plan_id:
        value = plans.get(plan_id)
        if not value:
            raise ValueError(f"No active plan `{plan_id}` exists in this conversation.")
        return agent_orchestrator.OrchestratorPlan.from_dict(value)
    active = [agent_orchestrator.OrchestratorPlan.from_dict(value)
              for value in plans.values()
              if value.get("status") not in {"executed", "cancelled", "expired"}]
    if len(active) != 1:
        if not active:
            raise ValueError("There is no active orchestration plan in this conversation.")
        raise ValueError("Multiple plans are active. Approve one explicitly with `approve plan PLAN_ID`.")
    return active[0]


def _save_orchestrator_plan(ctx, state, plan):
    plans = state.setdefault("orchestrator_plans", {})
    plans[plan.plan_id] = plan.to_dict()
    state["active_orchestrator_plan_id"] = plan.plan_id
    _save_state(ctx, state)


def _render_orchestrator_slack_message(plan):
    """Render the final Slack API text payload for an orchestration plan.

    This is the sole plan-preview presentation boundary.  It emits Slack
    mrkdwn directly; no Markdown or emoji-image conversion occurs afterward.
    """
    context_type = plan.context.get("context_type")
    relevant_count = plan.context.get("relevant_count", 0)
    lines = ["*Execution Plan*", "", "*Goal*",
             slack_presentation.text(plan.goal), "", "*Current State*",
             (f"• {relevant_count} overdue pending task{'s' if relevant_count != 1 else ''}"
              if context_type == "overdue_work" and relevant_count else
              "• No overdue pending tasks found."
              if context_type == "overdue_work" else
              f"• {relevant_count} relevant pending task{'s' if relevant_count != 1 else ''}"),
             f"• {plan.context.get('p1_count', 0)} P1 task{'s' if plan.context.get('p1_count', 0) != 1 else ''}",
             f"• {plan.context.get('unassigned_count', 0)} unassigned",
             f"• {plan.context.get('risk_count', 0)} active risk signal{'s' if plan.context.get('risk_count', 0) != 1 else ''}"]
    task_rows = plan.context.get("task_rows") or []
    if task_rows:
        heading = "*Overdue Work*" if context_type == "overdue_work" else "*Relevant Work*"
        lines.extend(("", heading))
        for row in task_rows:
            lines.append(
                f"• *{slack_presentation.text(row['name'])}* · "
                f"{slack_presentation.text(row['priority'])} · "
                f"{slack_presentation.text(row['owner'])} · due {slack_presentation.text(row['due'])}")
    lines.extend(("", "*Prepared Actions*"))
    visible_steps = [step for step in plan.steps if step.status != "removed"]
    displayed_steps = visible_steps[:10]
    if not displayed_steps:
        lines.append("• No action is currently required.")
    for index, step in enumerate(displayed_steps, 1):
        label = {
            "send_reminder": "Prepare reminder for",
            "send_deadline_reminder": "Prepare deadline reminder for",
            "review_owner_assignment": "Review ownership for",
            "assign_owner": "Review ownership for",
            "review_workload": "Review workload for",
            "review_task": "Review status for",
        }.get(step.action_type, step.action_type.replace("_", " ").title())
        suffix = "" if step.authorized else " · not authorized for execution"
        lines.append(f"{index}. {label} *{slack_presentation.text(step.target)}*{suffix}")
    if len(visible_steps) > len(displayed_steps):
        remaining = len(visible_steps) - len(displayed_steps)
        lines.append(f"• {remaining} additional prepared action{'s are' if remaining != 1 else ' is'} stored in this plan.")
    team_insights = plan.context.get("team_insights") or []
    if team_insights:
        lines.extend(("", "*Team Insight*"))
        lines.extend(slack_presentation.text(insight['message'])
                     for insight in team_insights)
    levels = ({step.risk_level for step in visible_steps}
              | {insight.get("risk_level") for insight in team_insights})
    risk = "🔴 High" if "critical" in levels else "🟠 Medium" if levels else "🟢 Low"
    lines.extend(("", "*Risk*", risk, ""))
    approval_count = sum(step.requires_approval and step.authorized for step in visible_steps)
    if plan.approval_status == "required":
        lines.extend(("*Approval Required*",
                      f"{approval_count} action{'s are' if approval_count != 1 else ' is'} ready for approval.",
                      "", f"`approve plan {plan.plan_id}`", "",
                      "*No actions have been taken.*"))
    else:
        lines.extend(("*Approval*", "No executable actions require approval.", "",
                      "*No changes were made.*"))
    rendered = "\n".join(lines)
    logger.info("orchestrator_renderer format=slack_mrkdwn sections=%d chars=%d",
                sum(line.startswith("*") and line.endswith("*") for line in lines),
                len(rendered))
    return rendered


def _resolve_orchestrator_context(context_type, snapshot, analysis_schema, risks):
    """Select authorized tasks using existing deterministic intelligence results."""
    today = current_date()
    pending = [task for task in snapshot if not task.completed]
    risk_entries = project_intelligence.analyze_task_risks(snapshot, analysis_schema, today)
    reasons_by_id = {entry.item_id: entry.reasons for entry in risk_entries}
    sentinel_ids = {task_id for risk in risks for task_id in risk.task_ids}
    if context_type == "overdue_work":
        selected = [task for task in pending if any(
            reason.startswith("Overdue by") for reason in reasons_by_id.get(task.item_id, ()))]
    elif context_type == "due_today":
        selected = [task for task in pending if "Due today" in reasons_by_id.get(task.item_id, ())]
    elif context_type == "due_24h":
        selected = [task for task in pending if task.due_date and today <= task.due_date <= today + timedelta(days=1)]
    elif context_type == "due_48h":
        selected = [task for task in pending if task.due_date and today <= task.due_date <= today + timedelta(days=2)]
    elif context_type == "deadline_risk":
        ids = {task_id for risk in risks if risk.risk_type in {
            "deadline_risk", "unassigned_deadline_risk"} for task_id in risk.task_ids}
        selected = [task for task in pending if task.item_id in ids]
    elif context_type == "priority_p1":
        selected = [task for task in pending if task.priority == "P1"]
    elif context_type == "unassigned_tasks":
        selected = [task for task in pending if not task.owner_ids]
    elif context_type == "current_risks":
        selected = [task for task in pending if task.item_id in sentinel_ids]
    elif context_type == "team_workload":
        selected = pending
    else:
        return None
    return sorted(selected, key=lambda task: (task.due_date or date.max, task.name.casefold()))


def _execute_orchestrator_plan(plan, ctx, state):
    if plan.requester_id != ctx.user_id:
        raise PermissionError("Only the user who created this plan may approve it.")
    if plan.status == "executed":
        logger.info("orchestrator_idempotency plan_id=%s duplicate=true", plan.plan_id)
        return ("*Plan Already Processed*\n\n"
                "This plan has already been executed or handled.\n\n"
                "No duplicate actions were performed.")
    logger.info("orchestrator_idempotency plan_id=%s duplicate=false", plan.plan_id)
    if plan.expires_at < time.time():
        expired = replace(plan, status="expired", approval_status="expired")
        _save_orchestrator_plan(ctx, state, expired)
        raise ValueError("*Plan Expired*\n\nThis plan has expired. Please regenerate it.\n\nNo actions were taken.")
    schema, items, visible, _ = _authorized_read_items({"intent": "list"}, ctx, default_pending=False)
    by_id = {slack_tools.extract_item_id(item): item for item in visible}
    member_names = {member.get("id"): member.get("name")
                    for member in slack_tools.list_workspace_members() if member.get("id")}
    updated_steps = []
    stale = []
    for step in plan.steps:
        if not step.executable or not step.authorized:
            updated_steps.append(step)
            continue
        item = by_id.get(step.target_task_id)
        if (not item or workflow_safety.item_fingerprint(item, schema) != step.task_fingerprint):
            stale.append(step.target)
        updated_steps.append(step)
    if stale:
        logger.info("orchestrator_stale_plan plan_id=%s stale_steps=%d", plan.plan_id, len(stale))
        raise ValueError("*Plan Expired*\n\nThe plan is no longer valid because the underlying task state changed.\n\nNo actions were executed.")
    completed, skipped = [], []
    for step in updated_steps:
        if not step.executable or not step.authorized:
            continue
        item = by_id[step.target_task_id]
        owners = tuple(slack_tools.extract_assignee_ids(item, schema))
        allowed = ((ctx.user_id in owners and config.has_permission(ctx, "update"))
                   or (ctx.user_id not in owners and config.has_permission(ctx, "update_others")))
        if not allowed or owners != step.target_user_ids:
            skipped.append((step.target, "authorization or ownership changed"))
            updated_steps[updated_steps.index(step)] = replace(
                step, status="skipped", execution_result="authorization changed")
            continue
        prepared = (step.prepared_message
                    or "Please confirm the current status or update the deadline.")
        message = (f"🔔 *Action Item Follow-up*\n\n"
                   f"{slack_presentation.text(prepared)}")
        try:
            for owner_id in owners:
                _send_deadline_reminder(owner_id, message)
            owner_names = ", ".join(member_names.get(owner_id) or user_name(owner_id)
                                    for owner_id in owners)
            completed.append((step.target, owner_names or "the task owner"))
            updated_steps[updated_steps.index(step)] = replace(
                step, status="verified", execution_result="reminder delivery accepted")
            logger.info("orchestrator_execution plan_id=%s step_id=%s action=%s",
                        plan.plan_id, step.step_id, step.action_type)
        except Exception:
            logger.exception("orchestrator_execution_failed plan_id=%s step_id=%s",
                             plan.plan_id, step.step_id)
            skipped.append((step.target, "reminder delivery failed"))
            updated_steps[updated_steps.index(step)] = replace(
                step, status="failed", execution_result="reminder delivery failed")
    finished = replace(
        plan, steps=tuple(updated_steps), status="executed", approval_status="approved",
        execution_status="verified" if completed and not skipped else "completed_with_skips")
    _save_orchestrator_plan(ctx, state, finished)
    logger.info("orchestrator_verification plan_id=%s success=%s", plan.plan_id, bool(completed))
    lines = (["*Plan Executed*", "", "*Completed*"] if not skipped else
             ["*Plan Not Completed*", "", "The plan could not be fully executed.", "", "*Completed*"])
    lines.extend(
        f"• Reminder sent to {slack_presentation.text(owner)} for *{slack_presentation.text(task)}*"
        for task, owner in completed)
    if not completed:
        lines.append("• None")
    if skipped:
        lines.extend(("", "*Failed*", *(
            f"• Reminder for *{slack_presentation.text(task)}* · {slack_presentation.text(reason)}"
            for task, reason in skipped), "", "Failed actions were not retried automatically."))
    lines.extend(("", "*Verification*",
                  "🟢 All approved actions verified." if completed and not skipped
                  else "Completed actions were verified; failed actions were not retried automatically."))
    return "\n".join(lines)


def handle_orchestrator(parsed, ctx, memory_key):
    """Coordinate planning and approved execution over existing trusted services."""
    mode = parsed.get("orchestrator_mode") or "create"
    state = _state(ctx)
    if mode == "create":
        schema, items, visible, _ = _authorized_read_items(
            {"intent": "list"}, ctx, default_pending=False)
        readable = _readable_analysis_schema(schema, ctx)
        snapshot = project_intelligence.normalize_task_snapshot(visible, readable)
        member_names = {member.get("id"): member.get("name")
                        for member in slack_tools.list_workspace_members() if member.get("id")}
        display_name = lambda user_id: member_names.get(user_id) or user_name(user_id)
        settings = action_item_sentinel.SentinelSettings.from_env()
        risks = action_item_sentinel.detect_risks(
            snapshot, current_date(), settings.warning_days,
            settings.combined_task_threshold)
        goal = parsed.get("goal") or "Review the current situation and prepare the next steps."
        context_type = agent_orchestrator.resolve_goal_context(goal)
        relevant = _resolve_orchestrator_context(
            context_type, snapshot, readable, risks)
        relevant_ids = ({task.item_id for task in relevant}
                        if relevant is not None else {task.item_id for task in snapshot})
        recommendations = {}
        for risk in risks:
            if len(risk.task_ids) != 1 or risk.task_ids[0] not in relevant_ids:
                continue
            task = next((value for value in snapshot if value.item_id == risk.task_ids[0]), None)
            if not task:
                continue
            payload = {
                "task_ids": list(risk.task_ids), "task_id": risk.task_id,
                "owner_ids": list(risk.owner_ids), "risk_type": risk.risk_type,
                "task_name": risk.task_name,
                "due_date": risk.due_date.isoformat() if risk.due_date else None,
                "task_state_hash": action_item_sentinel.task_state_hash(task),
            }
            recommendations[task.item_id] = smart_task_autopilot.prepare_recommendation(
                alert_id=f"orchestrator:{risk.risk_type}:{risk.task_id}", payload=payload,
                requesting_user=ctx.user_id,
                owner_name=", ".join(display_name(value) for value in risk.owner_ids) or None,
                today=current_date(), created_at=time.time())
        fingerprints = {slack_tools.extract_item_id(item): workflow_safety.item_fingerprint(item, schema)
                        for item in visible}
        plan = agent_orchestrator.build_plan(
            goal=goal, requester_id=ctx.user_id, tasks=snapshot, risks=risks,
            fingerprints=fingerprints, context_type=context_type,
            relevant_tasks=relevant, recommendations=recommendations)
        selected = list(relevant) if relevant is not None else [
            task for task in snapshot if task.item_id in set(plan.context["relevant_task_ids"])]
        task_rows = [{
            "name": task.name,
            "owner": ", ".join(display_name(value) for value in task.owner_ids) or "Unassigned",
            "priority": task.priority or "No priority",
            "due": (slack_presentation.compact_date(task.due_date.isoformat(), current_date())
                    if task.due_date else "No due date"),
        } for task in selected[:10]]
        plan = replace(plan, context={
            **plan.context,
            "p1_count": sum(task.priority == "P1" for task in selected),
            "unassigned_count": sum(not task.owner_ids for task in selected),
            "task_rows": task_rows,
        })

        def authorize(step):
            if step.action_type != "send_reminder":
                return True, None
            owners = step.target_user_ids
            allowed = ((ctx.user_id in owners and config.has_permission(ctx, "update"))
                       or (ctx.user_id not in owners and config.has_permission(ctx, "update_others")))
            return allowed, None if allowed else "Your role cannot send this owner follow-up."

        plan = agent_orchestrator.validate_plan(plan, authorize)
        _save_orchestrator_plan(ctx, state, plan)
        logger.info("orchestrator_request goal=%s actor_id=%s", context_type, ctx.user_id)
        logger.info("orchestrator_context_resolution goal=%s deterministic=true", context_type)
        logger.info("orchestrator_context context_type=%s records=%d source=normalized_snapshot",
                    context_type, len(selected))
        logger.info("orchestrator_plan_created plan_id=%s steps=%d", plan.plan_id, len(plan.steps))
        logger.info("orchestrator_validation plan_id=%s authorized=%d unauthorized=%d",
                    plan.plan_id, plan.validation_result.get("authorized", 0),
                    plan.validation_result.get("unauthorized", 0))
        return _render_orchestrator_slack_message(plan)
    plan = _orchestrator_plan(state, parsed.get("plan_id"))
    if mode == "approve":
        logger.info("orchestrator_approval plan_id=%s approved=true actor_id=%s",
                    plan.plan_id, ctx.user_id)
        return _execute_orchestrator_plan(plan, ctx, state)
    if mode == "cancel":
        cancelled = replace(plan, status="cancelled", approval_status="cancelled")
        _save_orchestrator_plan(ctx, state, cancelled)
        return f"Plan `{plan.plan_id}` cancelled. No actions were taken."
    if mode == "show_context":
        rows = plan.context.get("task_rows") or []
        if not rows:
            return f"*Plan {plan.plan_id} · Relevant Work*\n\nNo matching tasks are stored in this plan."
        return f"*Plan {plan.plan_id} · Relevant Work*\n\n" + "\n".join(
            f"• *{slack_presentation.text(row['name'])}* — "
            f"{slack_presentation.text(row['priority'])} — "
            f"{slack_presentation.text(row['owner'])} — {slack_presentation.text(row['due'])}"
            for row in rows)
    candidates = [step for step in plan.steps if step.action_type != "review_tasks"
                  and step.status not in {"removed", "verified"}]
    step_number = parsed.get("step_number")
    if step_number:
        if step_number > len(candidates):
            raise ValueError(f"This plan has only {len(candidates)} active prepared steps.")
        step = candidates[step_number - 1]
    elif len(candidates) != 1:
        raise ValueError("Which plan step do you mean? Please name the task or step explicitly.")
    else:
        step = candidates[0]
    if mode == "explain":
        evidence = " · ".join(step.evidence) or step.reason
        return (f"*Plan {plan.plan_id} · {slack_presentation.text(step.target)}*\n\n"
                f"Reason: {slack_presentation.text(evidence)}\n\nNo action has been taken.")
    removed = agent_orchestrator.update_step(
        plan, step.step_id, status="removed", executable=False,
        execution_result="removed by requester")
    _save_orchestrator_plan(ctx, state, removed)
    return f"Removed the step for *{slack_presentation.text(step.target)}*. No action was taken."


def _simulation_ledger():
    return decision_ledger.DecisionLedger(DB_PATH)


def _simulation_task_reference(reference, snapshot, state):
    """Resolve one hypothetical target only from the authorized snapshot."""
    if not reference or str(reference).casefold() in {"this", "that", "it", "this task", "that task"}:
        focus_ids = state.get("focus_ids") or []
        matches = [task for task in snapshot if task.item_id in focus_ids]
        if len(matches) == 1:
            return matches[0]
        raise ValueError("Please name the task you want to simulate.")
    wanted = normalize_task_name(reference)
    exact = [task for task in snapshot if normalize_task_name(task.name) == wanted]
    if len(exact) == 1:
        return exact[0]
    candidates = [task for task in snapshot if (
        wanted in normalize_task_name(task.name)
        or workflow_safety.title_similarity(reference, task.name) >= 0.72)]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = ", ".join(task.name for task in candidates[:5])
        raise ValueError(f"I found multiple matching tasks: {names}. Please name one task.")
    raise ValueError(f"I couldn't find an authorized task matching {reference!r}.")


def _simulation_targets(request, snapshot, state, ctx, schema):
    operation = request["operation"]
    parameters = {}
    if operation == "leave_unchanged":
        return [], parameters
    if operation == "workload_redistribution":
        names = {member["id"]: member["name"] for member in slack_tools.list_workspace_members()}
        report = project_intelligence.calculate_workload(
            snapshot, schema, lambda user_id: names.get(user_id) or user_name(user_id),
            current_date(), names)
        if not report.suggestions:
            raise ValueError("No deterministic workload redistribution is supported by the current authorized data.")
        suggestion = report.suggestions[0]
        task = next(value for value in snapshot if value.item_id == suggestion["item_id"])
        parameters["assignee_ids"] = [suggestion["to_user_id"]]
        return [task], parameters
    has_selector = any((
        request.get("target_unassigned"), request.get("target_priority"),
        request.get("target_overdue"), request.get("target_owner_name"),
        request.get("target_owner_self"), request.get("target_due"),
    ))
    if has_selector:
        today = current_date()
        matches = [task for task in snapshot if not task.completed]
        if request.get("target_unassigned"):
            matches = [task for task in matches if not task.owner_ids]
        if request.get("target_priority"):
            matches = [task for task in matches
                       if task.priority == request["target_priority"]]
        if request.get("target_overdue"):
            matches = [task for task in matches
                       if task.due_date and task.due_date < today]
        owner_id = None
        if request.get("target_owner_self"):
            owner_id = ctx.user_id
        elif request.get("target_owner_name"):
            owner_id = slack_tools.find_user_id(request["target_owner_name"])
            if not owner_id:
                raise ValueError(
                    f"I couldn't resolve the Slack member {request['target_owner_name']!r}.")
        if owner_id:
            matches = [task for task in matches if owner_id in task.owner_ids]
        target_due = request.get("target_due")
        if target_due == "today":
            matches = [task for task in matches if task.due_date == today]
        elif target_due == "tomorrow":
            matches = [task for task in matches
                       if task.due_date == today + timedelta(days=1)]
        elif target_due == "within_48h":
            matches = [task for task in matches if task.due_date
                       and today <= task.due_date <= today + timedelta(days=2)]
        if not matches:
            raise ValueError("No authorized pending tasks match that scenario selector.")
        if len(matches) > 1 and not request.get("selector_plural"):
            names = ", ".join(task.name for task in matches[:5])
            raise ValueError(
                f"I found {len(matches)} matching tasks: {names}. Please name exactly one task.")
        tasks = matches
    else:
        tasks = [_simulation_task_reference(request.get("task_reference"), snapshot, state)]
    if operation in {"assign_task", "reassign_task"}:
        assignee_names = request.get("assignee_names") or []
        assignee_ids = []
        for name in assignee_names:
            user_id = ctx.user_id if str(name).casefold() in {"me", "myself"} else slack_tools.find_user_id(name)
            if not user_id:
                raise ValueError(f"I couldn't resolve the Slack member {name!r}.")
            assignee_ids.append(user_id)
        parameters["assignee_ids"] = list(dict.fromkeys(assignee_ids))
    elif operation == "change_due_date":
        due = request.get("due_date")
        if request.get("due_date_offset_days") is not None:
            parameters["due_date_offset_days"] = int(request["due_date_offset_days"])
        else:
            parameters["due_date"] = due.isoformat() if isinstance(due, date) else str(due)
    elif operation == "change_priority":
        parameters["priority"] = request.get("priority")
    elif operation == "complete_task":
        parameters["completed"] = True
    return tasks, parameters


def _simulation_metric_lines(metrics, member_ids=(), name_for_user=user_name):
    lines = [
        f"• Pending: {metrics['pending']}",
        f"• Overdue: {metrics['overdue']}",
        f"• Unassigned: {metrics['unassigned']}",
        f"• P1: {metrics['priorities'].get('P1', 0)}",
    ]
    for user_id in member_ids:
        row = metrics["owners"].get(user_id, {})
        lines.append(
            f"• {slack_presentation.text(name_for_user(user_id))}: "
            f"{row.get('pending', 0)} pending · {row.get('p1', 0)} P1 · "
            f"{row.get('overdue', 0)} overdue")
    return lines


def _render_simulation(result, snapshot, name_for_user=user_name):
    selected = [task for task in snapshot if task.item_id in set(result.source_task_ids)]
    owner_ids = tuple(dict.fromkeys(result.parameters.get("assignee_ids") or ()))
    scenario = result.goal.rstrip(".?!") + "."
    metric_rows = lambda metrics: [
        ("Pending", metrics["pending"]),
        ("Overdue", metrics["overdue"]),
        ("Unassigned", metrics["unassigned"]),
        ("P1", metrics["priorities"].get("P1", 0)),
    ]
    lines = ["*🔎 Scenario Simulation*", "", "*What-if simulation — no changes made*", "",
             f"*Scenario · `{result.scenario_id}`*", scenario,
             "", "*Current State*", slack_presentation.render_slack_table(
                 ("Metric", "Value"), metric_rows(result.baseline_metrics)),
             "", "*Projected State*", slack_presentation.render_slack_table(
                 ("Metric", "Value"), metric_rows(result.simulated_metrics))]
    if selected:
        lines.extend(("", f"*Affected tasks · {len(selected)}*"))
        for task in selected:
            detail = []
            if result.operation == "change_due_date":
                current_due = task.due_date.isoformat() if task.due_date else "No due date"
                if result.parameters.get("due_date_offset_days") is not None:
                    projected_due = ((task.due_date + timedelta(
                        days=int(result.parameters["due_date_offset_days"]))).isoformat()
                                     if task.due_date else "No due date")
                else:
                    projected_due = result.parameters.get("due_date") or "No due date"
                detail.append(f"{current_due} → {projected_due}")
            detail.append(task.priority or "No priority")
            owners = ", ".join(name_for_user(owner) for owner in task.owner_ids) or "Unassigned"
            detail.append(owners)
            lines.append(
                f"• *{slack_presentation.text(task.name)}* — "
                f"{slack_presentation.text(' · '.join(detail))}")
        if result.operation == "change_due_date":
            destination = (f"by {result.parameters['due_date_offset_days']} days"
                           if result.parameters.get("due_date_offset_days") is not None
                           else f"to {result.parameters.get('due_date')}")
            lines.extend(("", "*Projected changes*",
                          f"• {len(selected)} due date{'s' if len(selected) != 1 else ''} would move {destination}",
                          "• Priority and assignee values would remain unchanged",
                          "• No Slack List records were modified"))
    lines.extend(("", "*Positive Impact*"))
    lines.extend(f"• {slack_presentation.text(value)}" for value in result.positive)
    if not result.positive:
        lines.append("• No measurable improvement in the modeled metrics.")
    lines.extend(("", "*Trade-offs*"))
    lines.extend(f"• {slack_presentation.text(value)}" for value in result.tradeoffs)
    if not result.tradeoffs:
        lines.append("• No additional measurable trade-off was introduced.")
    lines.extend(("", "*Risk Impact*",
                  f"• {len(result.impact['risk_removed'])} risk signals removed",
                  f"• {len(result.impact['risk_introduced'])} risk signals introduced"))
    lines.extend(("", "*Assumptions*"))
    lines.extend(f"• {slack_presentation.text(value)}" for value in result.assumptions)
    lines.extend(("", f"*Decision · `{result.decision_id}`*",
                  "No changes were made.",
                  "Reply `prepare this scenario` to create a separately approved change proposal."))
    return "\n".join(lines)


def _scenario_changes(result):
    operation = result.operation
    if operation in {"assign_task", "reassign_task", "workload_redistribution"}:
        value = result.parameters.get("assignee_ids") or []
        return [{"field": "assignee", "value": value[0] if len(value) == 1 else value}]
    if operation == "change_due_date":
        if result.parameters.get("due_date_offset_days") is not None:
            return []
        return [{"field": "due_date", "value": result.parameters["due_date"]}]
    if operation == "change_priority":
        return [{"field": "priority", "value": result.parameters["priority"]}]
    if operation == "complete_task":
        return [{"field": "completed", "value": True}]
    return []


def _prepare_simulation(result, ctx, snapshot, raw_items, schema):
    if result.requester_id != ctx.user_id:
        raise PermissionError("Only the user who created this simulation may prepare it.")
    if result.expires_at < time.time() or task_simulation.is_stale(result, snapshot):
        logger.info("decision_stale decision_id=%s actor_id=%s", result.decision_id, ctx.user_id)
        raise ValueError("Scenario is stale because the task state changed. Please run a new simulation.")
    changes = _scenario_changes(result)
    offset_days = result.parameters.get("due_date_offset_days")
    if not changes and offset_days is None:
        raise ValueError("This scenario has no consequential change to prepare.")
    by_id = {slack_tools.extract_item_id(item): item for item in raw_items}
    snapshot_by_id = {task.item_id: task for task in snapshot}
    entries = []
    for task_id in result.source_task_ids:
        if task_id not in by_id:
            continue
        task_changes = changes
        if offset_days is not None:
            task = snapshot_by_id.get(task_id)
            if not task or not task.due_date:
                raise ValueError("Scenario is stale because a task due date is no longer available.")
            task_changes = [{"field": "due_date", "value": (
                task.due_date + timedelta(days=int(offset_days))).isoformat()}]
        entries.append({"item_id": task_id, "item": by_id[task_id], "changes": task_changes})
    if len(entries) != len(result.source_task_ids):
        raise ValueError("Scenario is stale because a task no longer exists. Please run a new simulation.")
    # Existing proposal execution re-fetches state, verifies fingerprints, and
    # enforces mutation RBAC before the first write.
    _proposal_state(ctx, "simulation", entries, schema, {
        "decision_id": result.decision_id, "scenario_id": result.scenario_id,
        "expected_metrics": result.simulated_metrics,
    })
    _simulation_ledger().mark_prepared(
        result.decision_id, ctx.user_id, ctx.list_id, result.scenario_id)
    logger.info("decision_prepared decision_id=%s status=proposal_prepared", result.decision_id)
    lines = ["*Scenario Prepared — Approval Required*", "",
             f"Decision `{result.decision_id}` is ready as a validated proposal."]
    for task in [task for task in snapshot if task.item_id in set(result.source_task_ids)]:
        lines.append(f"• *{slack_presentation.text(task.name)}*")
    lines.extend(("", "No changes were made.",
                  "Reply `apply this proposal` to run fresh RBAC and stale-state checks, or `cancel`."))
    return "\n".join(lines)


def handle_simulation(parsed, ctx, memory_key):
    """Model authorized task state without invoking a mutation API."""
    mode = parsed.get("simulation_mode") or "create"
    ledger, state = _simulation_ledger(), _state(ctx)
    if mode == "history":
        rows = ledger.recent(ctx.user_id, ctx.list_id, failed_only=bool(parsed.get("failed_only")))
        if not rows:
            return "*Decision History*\n\nNo matching simulations are available."
        return "*Decision History*\n\n" + "\n".join(
            f"• `{row['decision_id']}` · {slack_presentation.text(row['scenario'].goal)} · "
            f"{row['execution_status']} / {row['verification_status']}" for row in rows)
    decision_id = parsed.get("decision_id") or state.get("active_simulation_decision_id")
    if mode in {"show_decision", "verify_decision", "show_scenario"}:
        record = (ledger.get_scenario(parsed.get("scenario_id"), ctx.user_id, ctx.list_id)
                  if mode == "show_scenario" else
                  ledger.get(decision_id, ctx.user_id, ctx.list_id) if decision_id else None)
        if not record:
            raise ValueError("I couldn't find an authorized simulation or decision with that ID.")
        result = record["scenario"]
        decision_id = result.decision_id
        if mode == "verify_decision":
            if not record["actual"]:
                return (f"*Decision Verification · `{decision_id}`*\n\n"
                        "This decision has not been executed, so no actual outcome is available.")
            status = "Verified" if record["verification_status"] == "verified" else "Variance Detected"
            return (f"*Decision Verification · `{decision_id}`*\n\n"
                    f"Expected pending: {result.simulated_metrics['pending']}\n"
                    f"Actual pending: {record['actual']['pending']}\n\n*Result:* {status}")
        return (f"*Decision · `{decision_id}`*\n\n{slack_presentation.text(result.goal)}\n\n"
                f"Status: {record['execution_status']} / {record['verification_status']}\n\n"
                "The expected outcome remains frozen in the decision ledger.")
    schema, raw_items, visible, _ = _authorized_read_items(
        {"intent": "list"}, ctx, default_pending=False)
    readable = _readable_analysis_schema(schema, ctx)
    snapshot = project_intelligence.normalize_task_snapshot(visible, readable)
    member_names = {member["id"]: member["name"]
                    for member in slack_tools.list_workspace_members() if member.get("id")}
    display_name = lambda user_id: member_names.get(user_id) or user_name(user_id)
    if mode == "prepare":
        active_id = state.get("active_simulation_decision_id")
        record = ledger.get(active_id, ctx.user_id, ctx.list_id) if active_id else None
        if not record:
            raise ValueError("There is no active simulation in this conversation.")
        return _prepare_simulation(record["scenario"], ctx, snapshot, raw_items, schema)
    request = deepcopy(parsed.get("scenario") or {})
    if request.get("target_priority") and not config.can_read_field(ctx, "priority"):
        raise PermissionError("Your role cannot use priority data for this simulation.")
    if ((request.get("target_overdue") or request.get("target_due")
         or request.get("operation") == "change_due_date")
            and not config.can_read_field(ctx, "due_date")):
        raise PermissionError("Your role cannot use due-date data for this simulation.")
    logger.info("simulation_request actor_id=%s list_id=%s mode=%s", ctx.user_id, ctx.list_id, mode)
    logger.info("simulation_intent operation=%s llm_used=false llm_call_count=0", request.get("operation"))
    if mode == "compare":
        results = []
        for assignee in request.get("assignee_names") or ():
            variant = {**request, "assignee_names": [assignee], "compare": False,
                       "goal": f"Assign {request.get('task_reference') or 'the selected task'} to {assignee}"}
            targets, parameters = _simulation_targets(variant, snapshot, state, ctx, readable)
            result = task_simulation.simulate(
                requester_id=ctx.user_id, goal=variant["goal"], operation=variant["operation"],
                tasks=snapshot, task_ids=[task.item_id for task in targets], parameters=parameters,
                today=current_date())
            results.append(ledger.create(result, ctx.list_id))
        state["simulation_comparison_ids"] = [result.decision_id for result in results]
        _save_state(ctx, state)
        option_names = [display_name(result.parameters["assignee_ids"][0]) for result in results]
        comparison_rows = []
        metrics = (
            ("Pending workload", lambda result: result.impact["owner_delta"]["pending"]),
            ("P1 workload", lambda result: result.impact["owner_delta"]["p1"]),
            ("Unassigned", lambda result: result.impact["unassigned_delta"]),
            ("New risk signals", lambda result: len(result.impact["risk_introduced"])),
        )
        for label, getter in metrics:
            comparison_rows.append((label, *(f"{getter(result):+d}" for result in results)))
        return slack_presentation.join_sections(
            "*🔎 Scenario Comparison*",
            slack_presentation.render_slack_table(
                ("Metric", *(slack_presentation.text(value) for value in option_names)),
                comparison_rows),
            slack_presentation.render_section(
                "Trade-offs", "The table shows objective modeled differences between the options."),
            "No option was automatically selected and no changes were made.")
    targets, parameters = _simulation_targets(request, snapshot, state, ctx, readable)
    result = task_simulation.simulate(
        requester_id=ctx.user_id, goal=request.get("goal") or "Task scenario",
        operation=request["operation"], tasks=snapshot,
        task_ids=[task.item_id for task in targets], parameters=parameters,
        today=current_date())
    result = ledger.create(result, ctx.list_id)
    state["active_simulation_decision_id"] = result.decision_id
    _save_state(ctx, state)
    logger.info("simulation_created scenario_id=%s decision_id=%s operation=%s",
                result.scenario_id, result.decision_id, result.operation)
    logger.info("simulation_projection scenario_id=%s task_count=%d risk_removed=%d risk_introduced=%d",
                result.scenario_id, len(result.source_task_ids), len(result.impact["risk_removed"]),
                len(result.impact["risk_introduced"]))
    logger.info("decision_created decision_id=%s status=%s", result.decision_id, result.status)
    return _render_simulation(result, snapshot, display_name)


def _proposal_state(ctx, kind, entries, schema, metadata=None):
    state = _state(ctx)
    state["proposal"] = {
        "kind": kind, "created": time.time(), "expires_at": time.time() + 1800,
        "entries": [{
            "item_id": entry["item_id"], "changes": deepcopy(entry["changes"]),
            "fingerprint": workflow_safety.item_fingerprint(entry["item"], schema),
        } for entry in entries],
        "metadata": metadata or {},
    }
    state["focus_ids"] = [entry["item_id"] for entry in entries]
    _save_state(ctx, state)


def handle_plan(parsed, ctx, memory_key):
    schema, _, relevant, assignee_ids = _authorized_read_items(parsed, ctx, default_pending=True)
    if not config.can_read_field(ctx, "due_date") or not config.can_read_field(ctx, "priority"):
        raise PermissionError("Planning requires readable due-date and priority fields.")
    period = parsed.get("planning_period") or {}
    try:
        start, end = date.fromisoformat(period["start"]), date.fromisoformat(period["end"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("I couldn't determine the planning period.")
    plan = project_intelligence.build_plan(relevant, schema, start, end, current_date())
    if not plan:
        return "*Suggested Plan*\n\nNo pending tasks match this planning request."
    entries = [{"item_id": entry.item_id, "item": entry.item,
                "changes": [{"field": "due_date", "value": entry.scheduled_date.isoformat()}]}
               for entry in plan]
    _proposal_state(ctx, "plan", entries, schema,
                    {"start": start.isoformat(), "end": end.isoformat(),
                     "resolved_assignee_ids": assignee_ids})
    by_day = {}
    for entry in plan:
        by_day.setdefault(entry.scheduled_date, []).append(entry)
    sections = ["*📅 Suggested Plan*", "_This is a proposal. No Slack List fields were changed._"]
    for day, day_entries in by_day.items():
        sections.append(f"*{day.strftime('%A')} — {day.isoformat()}*\n" + "\n".join(
            f"• *{slack_tools.extract_item_name(entry.item, schema)}* — {', '.join(entry.reasons)}"
            for entry in day_entries))
    sections.append("Reply *apply this plan* to update these exact tasks after a fresh RBAC and state check.")
    return "\n".join(sections)


def handle_workload(parsed, ctx):
    schema, _, relevant, _ = _authorized_read_items(parsed, ctx, default_pending=False)
    logger.info(
        "task_intelligence_started actor_id=%s list_id=%s task_count=%d operation=workload",
        ctx.user_id, ctx.list_id, len(relevant))
    if not config.can_read_field(ctx, "assignee"):
        raise PermissionError("Your role cannot read assignee data for workload analysis.")
    members = slack_tools.list_workspace_members()
    if not config.has_permission(ctx, "view_others"):
        members = [member for member in members if member["id"] == ctx.user_id]
    names = {member["id"]: member["name"] for member in members}
    def member_name(user_id):
        return names.get(user_id) or user_name(user_id)
    analysis_schema = _readable_analysis_schema(schema, ctx)
    report = project_intelligence.calculate_workload(
        relevant, analysis_schema, member_name, current_date(), names)
    if parsed.get("recommend_balance") and not config.has_permission(ctx, "view_others"):
        raise PermissionError("Your role cannot generate cross-member reassignment proposals.")
    if not report.rows or not any(row["pending"] for row in report.rows.values()):
        state = _state(ctx)
        state.update(last_intent="workload", visual_context={
            "kind": "workload", "response_mode": "text", "created": time.time()})
        _save_state(ctx, state)
        logger.info("visual_request intent=workload response_mode=text "
                    "visualization_type=none scope=authorized records=%d "
                    "llm_used=false llm_call_count=0", len(relevant))
        return "*👥 Team Workload* · No matching pending tasks found."
    active_rows = [(user_id, row) for user_id, row in report.rows.items() if row["pending"]]
    sorted_rows = sorted(
        active_rows, key=lambda pair: (-pair[1]["score"], pair[1]["name"].casefold()))
    lines = ["*📊 Team Workload*",
             slack_presentation.render_slack_table(
                 ("Owner", "Pending", "P1", "Overdue", "Due ≤48h"),
                 [(row["name"], row["pending"], row["p1"], row["overdue"], row["due_soon"])
                  for _, row in sorted_rows])]
    urgent_rows = [row for _, row in active_rows
                   if row["p1"] >= 2 or row["due_soon"] >= 2]
    if urgent_rows:
        lines.extend(("", "⚠️ *High Workload*"))
        lines.extend(
            f"• {slack_presentation.text(row['name'])} — {row['pending']} pending · "
            f"{row['due_soon']} due within 48h"
            for row in sorted(urgent_rows, key=lambda value: (-value["p1"], -value["due_soon"], value["name"].casefold())))
    if parsed.get("recommend_balance"):
        suggestions = report.suggestions
        if suggestions:
            entries = [{"item_id": suggestion["item_id"], "item": suggestion["item"],
                        "changes": [{"field": "assignee", "value": suggestion["to_user_id"]}]}
                       for suggestion in suggestions]
            _proposal_state(ctx, "workload_balance", entries, schema)
            lines.append("*Proposed balance*\n" + "\n".join(
                f"• Move *{slack_tools.extract_item_name(s['item'], schema)}* from "
                f"{slack_presentation.text(member_name(s['from_user_id']))} to "
                f"{slack_presentation.text(member_name(s['to_user_id']))} "
                f"because {s['reason']}." for s in suggestions))
            lines.append("No assignments were changed. Reply *apply this proposal* to run fresh RBAC and state checks.")
        else:
            lines.append("No safe rebalance suggestion is supported by the current workload data.")
    missing = []
    if not config.can_read_field(ctx, "due_date"):
        missing.append("overdue and upcoming counts")
    if not config.can_read_field(ctx, "priority"):
        missing.append("P1 counts")
    if missing:
        lines.append("*Data limitations*\n• Your role cannot read " + " or ".join(missing) + ".")
    logger.info(
        "team_workload_calculated actor_id=%s list_id=%s task_count=%d pending_count=%d "
        "member_count=%d llm_used=false llm_call_count=0",
        ctx.user_id, ctx.list_id, len(relevant),
        sum(row["pending"] for row in report.rows.values()), len(report.rows))
    state = _state(ctx)
    state.update(last_intent="workload", visual_context={
        "kind": "workload", "response_mode": "text", "created": time.time()})
    _save_state(ctx, state)
    logger.info("visual_request intent=workload response_mode=text visualization_type=none "
                "scope=authorized records=%d llm_used=false llm_call_count=0", len(relevant))
    return slack_presentation.join_sections(*lines)


def handle_similar_tasks(parsed, ctx, memory_key):
    """Describe title similarity and metadata differences without mutating tasks."""
    schema, items, visible, _ = _authorized_read_items(parsed, ctx, default_pending=False)
    logger.info(
        "task_intelligence_started actor_id=%s list_id=%s task_count=%d operation=similarity",
        ctx.user_id, ctx.list_id, len(visible))
    baseline = None
    baseline_name = parsed.get("task_name")
    if not baseline_name or baseline_name == "__LAST__":
        source = resolve_target_set(parsed, items, schema, memory_key, ctx, "inspect")
        if len(source.items) != 1:
            raise ValueError("I couldn't identify one task to compare.")
        baseline = source.items[0]
        baseline_name = slack_tools.extract_item_name(baseline, schema)
    baseline_id = slack_tools.extract_item_id(baseline) if baseline else None
    visible_ids = {slack_tools.extract_item_id(item) for item in visible}
    candidates = []
    for item in items:
        item_id = slack_tools.extract_item_id(item)
        if item_id == baseline_id or item_id not in visible_ids:
            continue
        score = workflow_safety.title_similarity(
            baseline_name, slack_tools.extract_item_name(item, schema))
        if score >= 0.65:
            candidates.append((score, item))
    candidates.sort(key=lambda pair: (-pair[0], slack_tools.extract_item_name(pair[1], schema).casefold()))
    logger.info(
        "task_similarity_analysis_completed actor_id=%s list_id=%s task_count=%d similar_count=%d "
        "llm_used=false llm_call_count=0",
        ctx.user_id, ctx.list_id, len(visible), len(candidates))
    if not candidates:
        return f"🔎 *Similar Tasks*\n\nNo similar tasks were found for *{slack_presentation.text(baseline_name)}*."
    shown = [item for _, item in candidates[:5]]
    store_view(memory_key, shown, schema, ctx, query_filter=deepcopy(parsed))
    lines = ["🔎 *Similar Tasks*", "", f"*Compared with:* {slack_presentation.text(baseline_name)}"]
    baseline_metadata = ({
        "Owner": item_assignees(baseline, schema) or "Unassigned",
        "Priority": slack_tools.extract_priority(baseline, schema) or "No priority",
        "Due": slack_presentation.compact_date(slack_tools.extract_due_date(baseline, schema), current_date()),
    } if baseline else None)
    for score, item in candidates[:5]:
        name = slack_presentation.text(slack_tools.extract_item_name(item, schema))
        metadata = {
            "Owner": item_assignees(item, schema) or "Unassigned",
            "Priority": slack_tools.extract_priority(item, schema) or "No priority",
            "Due": slack_presentation.compact_date(slack_tools.extract_due_date(item, schema), current_date()),
        }
        differences = ([f"{label}: {baseline_metadata[label]} → {value}"
                        for label, value in metadata.items() if baseline_metadata[label] != value]
                       if baseline_metadata else [f"{label}: {value}" for label, value in metadata.items()])
        label = "Exact duplicate title" if score == 1.0 else "Similar title"
        lines.append(f"\n• *{name}* — {label}")
        if differences:
            lines.append("  " + " · ".join(slack_presentation.text(value) for value in differences))
    lines.extend(("", "No tasks were merged or changed."))
    return "\n".join(lines)


def handle_standup(parsed, ctx, memory_key):
    schema, _, relevant, assignee_ids = _authorized_read_items(parsed, ctx, default_pending=False)
    analysis_schema = _readable_analysis_schema(schema, ctx)
    if not (config.can_read_field(ctx, "status") or config.can_read_field(ctx, "completed")):
        raise PermissionError("Your role cannot read task status for a standup.")
    report = project_intelligence.build_standup(relevant, analysis_schema, current_date())
    if slack_tools.column(schema, keys=slack_tools.DUE_KEYS, names={"Due Date", "Date"}) and not config.can_read_field(ctx, "due_date"):
        report.limitations.append("Due-date based attention and upcoming sections are unavailable for your role.")
    title = "🌅 Daily Standup"
    sections = [f"*{title}*"]
    completed_title = "Completed today" if report.completion_is_daily else "Completed (current List state)"
    sections.append(format_items(report.completed, schema, completed_title, ctx))
    sections.append(format_items(report.pending, schema, "Pending", ctx))
    sections.append(_health_response(report.attention, schema, ctx, "Needs Attention"))
    sections.append(format_items(report.upcoming, schema, "Due Tomorrow", ctx))
    if report.limitations:
        sections.append("*Data limitations*\n" + "\n".join(f"• {message}" for message in report.limitations))
    displayed = []
    for item in [*report.completed, *report.pending, *report.upcoming]:
        if slack_tools.extract_item_id(item) not in {slack_tools.extract_item_id(x) for x in displayed}:
            displayed.append(item)
    if displayed:
        store_view(memory_key, displayed, schema, ctx,
                   query_filter={**deepcopy(parsed), "resolved_assignee_ids": assignee_ids})
    return "\n".join(sections)


def _record_verified_audit(ctx, schema, item_id, operation, changes, before, after):
    try:
        audit_log.record(DB_PATH, ctx, item_id, operation, changes, schema, before, after)
    except Exception as exc:
        logger.warning("Verified mutation could not be written to audit history: %s", exc)


def handle_apply_proposal(ctx):
    state = _state(ctx)
    proposal = state.get("proposal") or {}
    if not proposal or proposal.get("expires_at", 0) < time.time():
        state.pop("proposal", None)
        _save_state(ctx, state)
        raise ValueError("There is no current proposal to apply in this conversation.")
    schema = slack_tools.get_list_schema(ctx.list_id)
    items = slack_tools.list_action_items(ctx, ctx.list_id)
    by_id = {slack_tools.extract_item_id(item): item for item in items}
    entries = proposal.get("entries") or []
    if len(entries) > 1 and not config.has_permission(ctx, "bulk"):
        raise PermissionError("Your role cannot apply a proposal that updates multiple action items.")
    for entry in entries:
        item = by_id.get(entry["item_id"])
        if not item or workflow_safety.item_fingerprint(item, schema) != entry["fingerprint"]:
            state.pop("proposal", None)
            _save_state(ctx, state)
            raise ValueError("The proposed tasks changed or no longer exist. Generate a new proposal; no changes were made.")
    prepared = []
    for entry in entries:
        changes = mutations.prepare_changes(
            {"intent": "update", "changes": entry["changes"]}, ctx, schema, current_date())
        mutations.authorize_collection([entry["item_id"]], items, "update", changes, ctx, schema)
        prepared.append((entry, changes))
    results = []
    for entry, changes in prepared:
        before = by_id[entry["item_id"]]
        result = mutations.execute(entry["item_id"], "update", changes, ctx, schema)
        results.append((entry, changes, result))
        if result.verified:
            _record_verified_audit(ctx, schema, entry["item_id"], "update", changes, before, result.item)
    state.pop("proposal", None)
    state["focus_ids"] = [entry["item_id"] for entry, _, _ in results]
    _save_state(ctx, state)
    lines = []
    for entry, _, result in results:
        name = slack_tools.extract_item_name(by_id[entry["item_id"]], schema)
        lines.append(f"• *{name}*: " + ("verified" if result.verified else "; ".join(result.problems)))
    heading = "Proposal applied and verified" if all(result.verified for _, _, result in results) else "Proposal results — not all changes verified"
    metadata = proposal.get("metadata") or {}
    if proposal.get("kind") == "simulation" and metadata.get("decision_id"):
        _, _, actual_visible, _ = _authorized_read_items(
            {"intent": "list"}, ctx, default_pending=False)
        readable = _readable_analysis_schema(schema, ctx)
        actual_snapshot = project_intelligence.normalize_task_snapshot(actual_visible, readable)
        actual_metrics = task_simulation.calculate_metrics(actual_snapshot, current_date())
        expected_metrics = metadata.get("expected_metrics") or {}
        verified = all(result.verified for _, _, result in results) and actual_metrics == expected_metrics
        logger.info("decision_approved decision_id=%s actor_id=%s",
                    metadata["decision_id"], ctx.user_id)
        _simulation_ledger().record_outcome(
            metadata["decision_id"], ctx.user_id, ctx.list_id, actual_metrics, verified)
        logger.info("decision_executed decision_id=%s verified=%s",
                    metadata["decision_id"], str(verified).lower())
        logger.info("%s decision_id=%s",
                    "decision_verified" if verified else "decision_variance",
                    metadata["decision_id"])
    return f"*{heading}*\n\n" + "\n".join(lines)


def handle_confirmation(ctx, key):
    state = _state(ctx)
    confirmation = state.get("confirmation") or {}
    if not workflow_safety.confirmation_is_fresh(confirmation):
        state.pop("confirmation", None)
        _save_state(ctx, state)
        raise ValueError("There is no current confirmation in this conversation, or it has expired.")
    schema = slack_tools.get_list_schema(ctx.list_id)
    items = slack_tools.list_action_items(ctx, ctx.list_id)
    ids = confirmation.get("item_ids") or []
    by_id = {slack_tools.extract_item_id(item): item for item in items}
    expected = confirmation.get("item_fingerprints") or {}
    stale = any(item_id not in by_id for item_id in ids) or any(
        workflow_safety.item_fingerprint(by_id[item_id], schema) != expected[item_id]
        for item_id in ids if item_id in by_id and item_id in expected)
    stale = stale or workflow_safety.snapshot_fingerprint(items, schema, ids) != confirmation.get("fingerprint")
    if stale:
        state.pop("confirmation", None)
        _save_state(ctx, state)
        raise ValueError(
            "Some tasks changed since the preview. Please review the updated changes. "
            "The affected tasks changed after confirmation was requested; no changes were made.")
    command = deepcopy(confirmation.get("command") or {})
    state.pop("confirmation", None)
    _save_state(ctx, state)
    if confirmation.get("kind") == "duplicate_create":
        command["_duplicate_approved"] = True
        return _dispatch(command, ctx, key)
    if confirmation.get("kind") == "bulk_mutation":
        command["target_ids"] = list(ids)
        command["_confirmation_approved"] = True
        return handle_mutation(command, ctx, key)
    if confirmation.get("kind") == "media_create":
        command["_media_approved"] = True
        return _dispatch(command, ctx, key)
    raise ValueError("The saved confirmation is not a supported operation.")


def handle_cancel(ctx):
    state = _state(ctx)
    had_pending = bool(state.pop("confirmation", None) or state.pop("proposal", None))
    _save_state(ctx, state)
    return "Cancelled. No Slack List changes were made." if had_pending else "There is no pending proposal or confirmation to cancel."


def handle_dependencies(parsed, ctx, memory_key):
    schema, items, relevant, _ = _authorized_read_items(parsed, ctx, default_pending=False)
    analysis_schema = _readable_analysis_schema(schema, ctx)
    fields = project_intelligence.dependency_fields(analysis_schema)
    if project_intelligence.dependency_fields(schema) and not fields:
        raise PermissionError("Your role cannot read dependency or blocker fields.")
    if not fields:
        return ("🔗 *Task Dependencies*\n\nNo known dependencies found for this task.\n\n"
                "_This Slack List has no explicit dependency or blocker field, so I cannot reliably determine "
                "task dependencies. No dependency data was inferred._")
    if parsed.get("dependency_origin"):
        origin_command = {"intent": "inspect", "task_name": parsed["dependency_origin"],
                          "target_scope": "single", "tasks": [], "changes": []}
        origin = resolve_target_set(origin_command, items, schema, memory_key, ctx, "inspect")
        if len(origin.items) != 1:
            raise ValueError("I couldn't resolve the dependency origin to one exact task.")
        origin_item = origin.items[0]
        origin_id = slack_tools.extract_item_id(origin_item)
        origin_name = slack_tools.extract_item_name(origin_item, schema)
        needles = {normalize_task_name(origin_id), normalize_task_name(origin_name)}
        dependents = []
        for item in relevant:
            values = project_intelligence.dependency_values(item, analysis_schema)
            normalized_values = " ".join(normalize_task_name(value) for value in values)
            if any(needle and needle in normalized_values for needle in needles):
                dependents.append((item, values))
        if not dependents:
            return f"No task explicitly lists *{origin_name}* as a dependency."
        store_view(memory_key, [item for item, _ in dependents], schema, ctx, query_filter=deepcopy(parsed))
        return (f"*Tasks affected by completing {origin_name}*\n\n" + "\n".join(
            f"• *{slack_tools.extract_item_name(item, schema)}* — explicit dependencies: {', '.join(values)}"
            for item, values in dependents)
            + "\n\nCompleting the named task would satisfy one explicit dependency; tasks with additional dependencies may remain blocked.")
    if reference_from(parsed) or parsed.get("task_name"):
        relevant = list(resolve_target_set(parsed, items, schema, memory_key, ctx, "inspect").items)
    blocked = [(item, project_intelligence.dependency_values(item, analysis_schema)) for item in relevant]
    blocked = [(item, values) for item, values in blocked if values]
    if not blocked:
        return "🔗 *Task Dependencies*\n\nNo known dependencies found for this task."
    store_view(memory_key, [item for item, _ in blocked], schema, ctx, query_filter=deepcopy(parsed))
    return "🔗 *Task Dependencies*\n\n" + "\n".join(
        f"• *{slack_tools.extract_item_name(item, schema)}* — {', '.join(values)}"
        for item, values in blocked)


def _sentinel_engine():
    return action_item_sentinel.ActionItemSentinel(
        action_item_sentinel.SentinelStore(DB_PATH))


def _visible_sentinel_alerts(engine, list_id, visible_ids):
    """Return only active alerts whose complete task set is visible to this actor."""
    allowed = set(visible_ids)
    result = []
    for alert in engine.store.alerts(list_id, status="active"):
        task_ids = set(alert.payload.get("task_ids") or [alert.task_id])
        if task_ids and task_ids.issubset(allowed):
            result.append(alert)
    return result


def _sentinel_alert_message(task, schema, ctx, reasons):
    name = slack_presentation.text(slack_tools.extract_item_name(task, schema) or "Action item")
    owner = slack_presentation.text(item_assignees(task, schema) or "Unassigned")
    priority = slack_tools.extract_priority(task, schema) or "No priority"
    due = slack_presentation.compact_date(slack_tools.extract_due_date(task, schema), current_date())
    why = "\n".join(f"• {slack_presentation.text(reason)}" for reason in reasons)
    return (f"*Action Item Sentinel*\n\n🔴 *Immediate Attention*\n\n*{name}*\n"
            f"{priority} · {owner} · {due}\n\n*Why*\n{why}\n\n"
            "*Recommended Action*\nConfirm completion status or update the deadline.")


def _sentinel_empty_response(snapshot):
    pending = sum(not task.completed for task in snapshot)
    detail = ("There are pending action items, but none currently require Sentinel attention."
              if pending else "No pending action items currently require Sentinel attention.")
    return f"*Action Item Sentinel*\n\n*No active risks found.*\n\n{detail}"


def _format_sentinel_risks(risks, alerts, requesting_user, *,
                           max_task_risks=8, max_workload_risks=4):
    """Render task risks and aggregate signals as distinct Slack-native concepts."""
    alert_by_key = {(alert.event_type.removeprefix("risk:"), alert.task_id): alert
                    for alert in alerts if alert.event_type.startswith("risk:")}
    visible_risks = [risk for risk in risks
                     if (risk.risk_type, risk.task_id) in alert_by_key]
    task_risks = [risk for risk in visible_risks if risk.risk_type != "combined_workload_risk"]
    workload_risks = [risk for risk in visible_risks if risk.risk_type == "combined_workload_risk"]
    overdue = [risk for risk in task_risks if risk.risk_type == "overdue"]
    deadline = [risk for risk in task_risks if risk.risk_type != "overdue"]
    displayed_alerts = []
    lines = ["*Action Item Sentinel*"]
    if not visible_risks:
        return "*Action Item Sentinel*\n\n*No active risks found.*", []
    remaining = max_task_risks

    def add_task_section(heading, values):
        nonlocal remaining
        shown = values[:remaining]
        if not shown:
            return
        lines.extend(("", heading))
        for risk in shown:
            alert = alert_by_key.get((risk.risk_type, risk.task_id))
            displayed_alerts.append(alert)
            owner = ", ".join(user_name(value) for value in risk.owner_ids) or "Unassigned"
            due = (slack_presentation.compact_date(risk.due_date.isoformat(), current_date())
                   if risk.due_date else "No due date")
            lines.extend(("", f"{len(displayed_alerts)}. *{slack_presentation.text(risk.task_name)}*",
                          f"   {slack_presentation.text(owner)} · {risk.priority or 'No priority'} · {due}",
                          "   _Why:_ " + " · ".join(
                              slack_presentation.text(reason) for reason in risk.reasons)))
        remaining -= len(shown)

    add_task_section("🔴 *Overdue*", overdue)
    add_task_section("🟠 *Deadline Risk*", deadline)

    shown_workload = workload_risks[:max_workload_risks]
    if shown_workload:
        lines.extend(("", "👥 *Workload Risk*"))
        for risk in shown_workload:
            owner = ", ".join(user_name(value) for value in risk.owner_ids) or "Unassigned"
            lines.append(f"• {slack_presentation.text(owner)} — " +
                         " · ".join(slack_presentation.text(reason) for reason in risk.reasons))

    omitted = max(0, len(task_risks) - max_task_risks) + max(
        0, len(workload_risks) - max_workload_risks)
    if omitted:
        lines.extend(("", f"_{omitted} additional risk signal{'s were' if omitted != 1 else ' was'} omitted._"))
    recommendation_alerts = []
    for risk in visible_risks:
        alert = alert_by_key.get((risk.risk_type, risk.task_id))
        if alert and alert not in recommendation_alerts:
            recommendation_alerts.append(alert)
    if recommendation_alerts:
        recommendations = [smart_task_autopilot.prepare_recommendation(
            alert_id=alert.alert_id, payload=alert.payload,
            requesting_user=requesting_user,
            owner_name=(", ".join(user_name(value) for value in alert.payload.get("owner_ids") or [])
                        or None),
            today=current_date(), created_at=alert.created_at, status=alert.status)
            for alert in recommendation_alerts]
        primary = next(
            (recommendation for recommendation in recommendations if recommendation.executable),
            recommendations[0])
        display_positions = {alert.alert_id: position
                             for position, alert in enumerate(displayed_alerts, 1)}
        approval_position = display_positions.get(primary.alert_id)
        logger.info(
            "autopilot_recommendation_prepared recommendation_id=%s alert_id=%s "
            "action_type=%s executable=%s llm_used=false llm_call_count=0",
            primary.recommendation_id, primary.alert_id, primary.action_type,
            str(primary.executable).lower())
        workload_recommendations = [value for value in recommendations
                                    if value.action_type == "review_workload"]
        lines.extend(("", "*Smart Task Autopilot*", "", "*Recommended Action*"))
        if not primary.executable and workload_recommendations:
            lines.extend(f"• {slack_presentation.text(value.recommendation)}"
                         for value in workload_recommendations)
        else:
            lines.append(slack_presentation.text(primary.recommendation))
        if primary.executable and primary.prepared_message:
            lines.extend(("", "*Prepared Message*",
                          f"> {slack_presentation.text(primary.prepared_message)}"))
        lines.extend(("", "*Approval*"))
        if primary.executable and approval_position:
            lines.append(f"`send sentinel alert {approval_position}`")
        else:
            lines.append("Human review is required; no automatic action is available.")
        if approval_position:
            lines.append(f"To dismiss it, reply `dismiss sentinel alert {approval_position}`.")
        if primary.executable and workload_recommendations:
            lines.extend(("", "*Workload Recommendations*",
                          *(f"• {slack_presentation.text(value.recommendation)}"
                            for value in workload_recommendations)))
        lines.extend(("", "*No action has been taken automatically.*"))
    return "\n".join(lines), displayed_alerts


def handle_sentinel(parsed, ctx, memory_key):
    """Read Sentinel findings or execute one explicitly approved reminder."""
    if not config.has_permission(ctx, "view"):
        raise PermissionError("Your role cannot view Action Item Sentinel alerts.")
    mode = parsed.get("sentinel_mode") or "risks"
    query = deepcopy(parsed)
    query.pop("sentinel_mode", None)
    query.pop("sentinel_action", None)
    query.pop("selection_index", None)
    schema, _, visible, assignee_ids = _authorized_read_items(
        query, ctx, default_pending=False)
    analysis_schema = _readable_analysis_schema(schema, ctx)
    snapshot = project_intelligence.normalize_task_snapshot(visible, analysis_schema)
    engine = _sentinel_engine()
    by_id = {slack_tools.extract_item_id(item): item for item in visible}

    if mode == "action":
        position = parsed.get("selection_index") or 0
        displayed = engine.store.resolve_display(
            ctx.list_id, ctx.channel_id, ctx.user_id, position)
        if not displayed:
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: That alert number was not shown to you in this conversation.")
        alert_id, displayed_at = displayed
        alert = engine.store.alert(alert_id)
        if not alert:
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: The Sentinel alert no longer exists.")
        if alert.list_id != ctx.list_id:
            raise PermissionError("That Sentinel alert is outside this Slack List.")
        if time.time() - displayed_at > engine.settings.approval_ttl_minutes * 60:
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: The approval has expired. Run `show sentinel alerts` again.")
        if alert.status != "active":
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: This Sentinel alert has already been handled.")
        task_ids = alert.payload.get("task_ids") or [alert.task_id]
        if len(task_ids) != 1:
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: Workload signals are advisory and cannot send a single-task reminder.")
        task = by_id.get(task_ids[0])
        if not task:
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: The task no longer exists or is not authorized for you.")
        if parsed.get("sentinel_action") == "dismiss":
            if not engine.store.dismiss(alert.alert_id, ctx.user_id):
                raise ValueError("*Sentinel Action Not Completed*\n\nReason: This Sentinel alert has already been handled.")
            logger.info("sentinel_action_rejected alert_id=%s actor_id=%s action=dismiss", alert.alert_id, ctx.user_id)
            return ("*Sentinel Action Completed*\n\n"
                    f"Alert dismissed for:\n*{slack_presentation.text(slack_tools.extract_item_name(task, schema))}*\n\n"
                    "No reminder was sent.")
        owners = slack_tools.extract_assignee_ids(task, schema)
        if not owners:
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: This task has no assigned owner.")
        if ctx.user_id not in owners and not config.has_permission(ctx, "update_others"):
            raise PermissionError("Your role cannot send reminders for another member's task.")
        if ctx.user_id in owners and not config.has_permission(ctx, "update"):
            raise PermissionError("Your role cannot approve task follow-ups.")
        expected_task_hash = alert.payload.get("task_state_hash")
        normalized = next((value for value in snapshot if value.item_id == task_ids[0]), None)
        if (not normalized or normalized.completed
                or expected_task_hash != action_item_sentinel.task_state_hash(normalized)):
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: The alert is no longer actionable because the task state has changed.")
        current_risks = action_item_sentinel.detect_risks(
            [normalized], current_date(), engine.settings.warning_days,
            engine.settings.combined_task_threshold)
        if not any(risk.risk_type == alert.event_type.removeprefix("risk:")
                   and risk.task_id == alert.task_id for risk in current_risks):
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: The alert is no longer relevant to the task's current state.")
        recommendation = smart_task_autopilot.prepare_recommendation(
            alert_id=alert.alert_id, payload=alert.payload,
            requesting_user=ctx.user_id,
            owner_name=item_assignees(task, schema) or None,
            today=current_date(), created_at=displayed_at, status=alert.status)
        if (not recommendation.executable
                or recommendation.target_user not in owners
                or not recommendation.prepared_message):
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: This recommendation requires human task review and has no executable reminder.")
        if not engine.store.claim_action(alert.alert_id, ctx.user_id, alert.state_hash):
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: This Sentinel alert has already been handled or is no longer current.")
        try:
            reminder = ("🔔 *Action Item Follow-up*\n\n" +
                        slack_presentation.text(recommendation.prepared_message))
            for owner_id in owners:
                _send_deadline_reminder(owner_id, reminder)
            engine.store.finish_action(alert.alert_id, "approved")
            logger.info("sentinel_action_approved alert_id=%s actor_id=%s recipient_count=%d",
                        alert.alert_id, ctx.user_id, len(owners))
            logger.info("autopilot_action_completed recommendation_id=%s alert_id=%s "
                        "actor_id=%s action_type=%s",
                        recommendation.recommendation_id, alert.alert_id,
                        ctx.user_id, recommendation.action_type)
        except Exception:
            engine.store.finish_action(alert.alert_id, "active")
            logger.exception("sentinel_action_failed alert_id=%s actor_id=%s", alert.alert_id, ctx.user_id)
            raise ValueError("*Sentinel Action Not Completed*\n\nReason: The reminder could not be sent. The alert remains active.")
        name = slack_presentation.text(slack_tools.extract_item_name(task, schema) or "Action item")
        owner = slack_presentation.text(item_assignees(task, schema) or "Unassigned")
        priority = slack_tools.extract_priority(task, schema) or "No priority"
        status = "Overdue" if alert.event_type == "risk:overdue" else "Deadline risk"
        return ("*Sentinel Action Completed*\n\nReminder sent for:\n"
                f"*{name}*\n\nOwner: {owner}\nPriority: {priority}\nStatus: {status}")

    evaluation = engine.evaluate(
        ctx.list_id, snapshot,
        datetime.now(ZoneInfo(os.getenv("DEADLINE_REMINDER_TIMEZONE", "Asia/Kathmandu"))),
        persist_snapshot=False)
    alerts = _visible_sentinel_alerts(engine, ctx.list_id, by_id)

    if mode == "explain":
        targets = resolve_target_set(parsed, visible, schema, memory_key, ctx, "inspect")
        if len(targets.items) != 1:
            raise ValueError("Please identify one task to explain.")
        task_id = slack_tools.extract_item_id(targets.items[0])
        risks = [risk for risk in evaluation.risks if task_id in risk.task_ids]
        if not risks:
            return "*Action Item Sentinel*\n\nNo current deterministic risk was identified for that task."
        return _sentinel_alert_message(targets.items[0], schema, ctx, risks[0].reasons)

    if mode == "alerts":
        if not evaluation.risks:
            return _sentinel_empty_response(snapshot)
        response, displayed = _format_sentinel_risks(
            evaluation.risks, alerts, ctx.user_id)
        state = _state(ctx)
        state.update(sentinel_alert_ids=[alert.alert_id for alert in displayed],
                     sentinel_list_id=ctx.list_id)
        _save_state(ctx, state)
        engine.store.save_display(
            ctx.list_id, ctx.channel_id, ctx.user_id,
            [alert.alert_id for alert in displayed], time.time())
        return response

    if not evaluation.risks:
        return _sentinel_empty_response(snapshot)
    response, displayed_alerts = _format_sentinel_risks(
        evaluation.risks, alerts, ctx.user_id)
    if displayed_alerts:
        state = _state(ctx)
        state.update(sentinel_alert_ids=[alert.alert_id for alert in displayed_alerts],
                     sentinel_list_id=ctx.list_id)
        _save_state(ctx, state)
        engine.store.save_display(
            ctx.list_id, ctx.channel_id, ctx.user_id,
            [alert.alert_id for alert in displayed_alerts], time.time())
    logger.info("sentinel_response_generated actor_id=%s list_id=%s risk_count=%d llm_used=false llm_call_count=0",
                ctx.user_id, ctx.list_id, len(evaluation.risks))
    return response


def handle_history(parsed, ctx, memory_key):
    """Render truthful, RBAC-scoped history from the persistent mutation audit."""
    if not config.has_permission(ctx, "view"):
        raise PermissionError("Your role cannot view action-item history.")
    schema = slack_tools.get_list_schema(ctx.list_id)
    current_items = slack_tools.list_action_items(ctx, ctx.list_id)
    item_ids = []
    if reference_from(parsed) or parsed.get("task_name"):
        targets = resolve_target_set(parsed, current_items, schema, memory_key, ctx, "inspect")
        item_ids = list(targets.item_ids)
    if not config.has_permission(ctx, "view_others"):
        own_ids = {slack_tools.extract_item_id(item) for item in current_items
                   if ctx.user_id in slack_tools.extract_assignee_ids(item, schema)}
        if item_ids and any(item_id not in own_ids for item_id in item_ids):
            raise PermissionError("Your role can only view history for tasks assigned to you.")
        item_ids = item_ids or list(own_ids)
    timezone = ZoneInfo("Asia/Kathmandu")
    today = current_date()
    since = None
    period_start = None
    if parsed.get("history_period") == "today":
        period_start = today
        since = datetime.combine(period_start, datetime.min.time(), timezone).timestamp()
    elif parsed.get("history_period") == "since_yesterday":
        period_start = today - timedelta(days=1)
        since = datetime.combine(period_start, datetime.min.time(), timezone).timestamp()
    elif parsed.get("history_period") == "since_last_check":
        since = _state(ctx).get("command_center_checked_at")
        if not since:
            return ("No previous Command Center check is available in this conversation. "
                    "Run `command center` first.")
    elif parsed.get("history_period") == "this_week":
        period_start = today - timedelta(days=today.weekday())
        since = datetime.combine(period_start, datetime.min.time(), timezone).timestamp()
    entries = audit_log.history(DB_PATH, ctx.list_id, item_ids, since=since)
    tracking_timestamp = audit_log.tracking_started(DB_PATH, ctx.list_id)
    tracking_date = datetime.fromtimestamp(tracking_timestamp, timezone).strftime("%b %d, %Y")
    limitation = (f"_History tracking started on {tracking_date}. "
                  "Changes before that date may not be available._")
    if not entries and not period_start:
        return "No verified mutation history is available for that request.\n\n" + limitation
    item_names = {slack_tools.extract_item_id(item): slack_tools.extract_item_name(item, schema)
                  for item in current_items}

    def history_item_name(entry):
        return (item_names.get(entry["item_id"])
                or (entry.get("after") or {}).get("name")
                or (entry.get("before") or {}).get("name")
                or "Action item")

    def display_value(field, value):
        if field == "due_date":
            return slack_presentation.compact_date(value, today) if value else "No due date"
        if field == "priority":
            return value or "No priority"
        if field == "completed":
            return "Completed" if value else "Pending"
        if field == "assignees":
            values = value if isinstance(value, list) else ([value] if value else [])
            return ", ".join(user_name(user_id) for user_id in values) or "Unassigned"
        return str(value if value not in {None, ""} else "Not set")

    def differences(entry):
        before, after = entry.get("before") or {}, entry.get("after") or {}
        result = []
        for field, label in (("name", "Task"), ("priority", "Priority"),
                             ("due_date", "Due"), ("assignees", "Owner"),
                             ("completed", "Status")):
            control = "assignee" if field == "assignees" else field
            if not config.can_read_field(ctx, control):
                continue
            if before.get(field) != after.get(field):
                result.append((field, label, before.get(field), after.get(field)))
        known = {"name", "priority", "due_date", "assignee", "owner", "completed", "status"}
        for change in entry.get("changes") or []:
            field = change.get("field")
            if field and field not in known and config.can_read_field(ctx, field):
                result.append((field, field.replace("_", " ").title(), None, change.get("value")))
        return result

    wanted_field = parsed.get("history_field")
    if wanted_field:
        changes = [(entry, diff) for entry in entries for diff in differences(entry)
                   if diff[0] == wanted_field]
        if parsed.get("history_value") and wanted_field == "assignees":
            wanted_id = slack_tools.find_user_id(parsed["history_value"])
            changes = [(entry, diff) for entry, diff in changes
                       if wanted_id and wanted_id in (diff[3] or [])]
        if not changes:
            return "No verified history is available for that field.\n\n" + limitation
        entry, diff = changes[0]
        when = datetime.fromtimestamp(entry["created"], timezone).strftime("%b %-d, %Y at %-I:%M %p")
        if parsed.get("history_previous"):
            return (f"*Previous {diff[1]} · {slack_presentation.text(history_item_name(entry))}*\n\n"
                    f"{slack_presentation.text(display_value(diff[0], diff[2]))}\n\n{limitation}")
        return (f"*{diff[1]} history · {slack_presentation.text(history_item_name(entry))}*\n\n"
                f"{slack_presentation.text(display_value(diff[0], diff[2]))} → "
                f"{slack_presentation.text(display_value(diff[0], diff[3]))}\n"
                f"{when} · {slack_presentation.text(user_name(entry['actor_id']))}\n\n{limitation}")

    grouped = {"Created": [], "Updated": [], "Reassigned": [], "Completed": [],
               "Reopened": [], "Became overdue": []}
    for entry in reversed(entries):
        name = slack_presentation.text(history_item_name(entry))
        actor = slack_presentation.text(user_name(entry["actor_id"]))
        stamp = datetime.fromtimestamp(entry["created"], timezone).strftime("%b %-d · %-I:%M %p")
        operation = entry["operation"]
        diffs = differences(entry)
        if operation == "create":
            grouped["Created"].append(f"• *{name}* — {stamp} · {actor}")
            continue
        if operation == "complete":
            grouped["Completed"].append(f"• *{name}* — completed by {actor} · {stamp}")
            continue
        if operation == "reopen":
            grouped["Reopened"].append(f"• *{name}* — {stamp} · {actor}")
            continue
        reassigned = [diff for diff in diffs if diff[0] == "assignees"]
        for field, label, earlier, later in reassigned:
            grouped["Reassigned"].append(
                f"• *{name}* — {display_value(field, earlier)} → {display_value(field, later)} · {stamp}")
        remaining = [diff for diff in diffs if diff[0] != "assignees"]
        if remaining:
            details = "; ".join(
                f"{label}: {display_value(field, earlier)} → {display_value(field, later)}"
                for field, label, earlier, later in remaining)
            grouped["Updated"].append(f"• *{name}* — {details} · {stamp} · {actor}")

    if period_start:
        allowed_ids = set(item_ids) if item_ids else {
            slack_tools.extract_item_id(item) for item in current_items}
        for item in current_items:
            if (slack_tools.extract_item_id(item) not in allowed_ids
                    or slack_tools.extract_completed(item, schema)):
                continue
            try:
                due = date.fromisoformat(str(slack_tools.extract_due_date(item, schema))[:10])
            except (TypeError, ValueError):
                continue
            became_overdue = due + timedelta(days=1)
            if period_start <= became_overdue <= today:
                name = slack_presentation.text(slack_tools.extract_item_name(item, schema))
                grouped["Became overdue"].append(
                    f"• *{name}* — due {slack_presentation.compact_date(due.isoformat(), today)}")

    title = "Changes Since Last Check" if parsed.get("history_period") == "since_last_check" else (
        "Changes Since Yesterday" if parsed.get("history_period") == "since_yesterday" else (
        "Changes This Week" if parsed.get("history_period") == "this_week" else (
        "Changes Today" if parsed.get("history_period") == "today" else "Task History")))
    icons = {"Created": "🆕", "Updated": "🔄", "Reassigned": "👤",
             "Completed": "✅", "Reopened": "↩️", "Became overdue": "⚠️"}
    sections = [f"*📋 {title} · Verified mutation history*"]
    if not any(grouped.values()):
        sections.extend(("", "No verified changes were recorded for this period."))
    for label, values in grouped.items():
        if values:
            sections.extend(("", f"*{icons[label]} {label}*", *values))
    sections.extend(("", limitation))
    return "\n".join(sections)


def handle_members(parsed, ctx):
    if not config.has_permission(ctx, "view"):
        raise PermissionError("Your role cannot view workspace-member information.")
    members = slack_tools.list_workspace_members()
    wanted_ids = set(parsed.get("resolved_member_ids") or [])
    requested_role = config.normalize_role(parsed.get("role")) if parsed.get("role") else None
    if wanted_ids:
        members = [member for member in members if member["id"] in wanted_ids]
    if requested_role:
        members = [member for member in members
                   if config.get_user_role(member["id"], ctx.team_id) == requested_role]
    if not members:
        return "I couldn't find a workspace member matching that request."
    return "*Workspace members*\n\n" + "\n".join(
        f"• *{slack_presentation.text(member['name'])}* — "
        f"{slack_presentation.text(config.get_user_role(member['id'], ctx.team_id))}"
        for member in members)


_process_lock = threading.RLock()


def _media_preview(items, title="Extracted action items"):
    lines = [f"*{title}* · {len(items)}", ""]
    for index, item in enumerate(items, 1):
        row = slack_presentation.TaskRow(
            name=item.title, assignee=item.assignee or "Unassigned",
            due_date=item.due_date, show_due=True,
            priority=item.priority or "No priority", status="Pending")
        line = slack_presentation.task_line(
            row, position=index if len(items) > 5 else None,
            bullet=len(items) <= 5, today=current_date())
        if item.evidence:
            line += f" · _Evidence: “{slack_presentation.text(item.evidence[:120])}”_"
        if item.confidence < 0.75:
            line += " · ⚠️ review"
        lines.append(line)
    return "\n".join(lines)


def _stage_media_create(ctx, command, items, schema, current_items):
    ids = [slack_tools.extract_item_id(item) for item in current_items]
    state = _state(ctx)
    state["confirmation"] = {
        "kind": "media_create", "created": time.time(),
        "expires_at": time.time() + workflow_safety.CONFIRMATION_TTL_SECONDS,
        "command": deepcopy(command), "item_ids": ids,
        "fingerprint": workflow_safety.snapshot_fingerprint(current_items, schema, ids),
    }
    _save_state(ctx, state)


def _resolve_media_actions(items, ctx):
    """Resolve each extracted action independently so one unsafe item is isolated."""
    logger.info("assignee_resolution_started source=shared_content actor_id=%s task_count=%d",
                ctx.user_id, len(items))
    resolved, unresolved = [], []
    for item in items:
        command = validate_command(action_item_extraction.workflow_command(item))
        try:
            command = _resolve_command_members(command, ctx)
        except ValueError as exc:
            unresolved.append((item, str(exc)))
            continue
        resolved.append((item, command))
    logger.info(
        "assignee_resolution_completed source=shared_content actor_id=%s resolved=%d unresolved=%d",
        ctx.user_id, len(resolved), len(unresolved))
    return resolved, unresolved


def _media_member_issues(unresolved):
    blocks = []
    for item, message in unresolved:
        if item.assignee:
            row = slack_presentation.TaskRow(
                item.title, due_date=item.due_date, priority=item.priority, status="Pending")
            blocks.append(slack_presentation.assignee_clarification(
                row, item.assignee, today=current_date()))
        else:
            blocks.append(
                f"*Clarification required*\n\n*{slack_presentation.text(item.title)}*\n\n"
                f"{slack_presentation.text(message)}")
    return "\n\n".join(blocks)


def _shared_content_heading(items, partial=False):
    count = len(items)
    if partial:
        return f"*Action Items Processed* · {count} task{'s' if count != 1 else ''}"
    source_types = {item.source_type for item in items}
    if source_types == {"audio"}:
        return f"*Audio Processed* 🎧\n\nExtracted *{count} action item{'s' if count != 1 else ''}*."
    if source_types == {"video"}:
        return f"*Video Processed*\n\nExtracted *{count} action item{'s' if count != 1 else ''}*."
    if source_types == {"transcript"}:
        return f"*Transcript Processed*\n\nExtracted *{count} action item{'s' if count != 1 else ''}*."
    return f"*Shared Content Processed* · {count} action item{'s' if count != 1 else ''}"


def _media_source_label(source_types):
    values = set(source_types or [])
    if values == {"audio"}:
        return "Audio", "🎧"
    if values == {"video"}:
        return "Video", "🎥"
    if values == {"transcript"}:
        return "Transcript", "📄"
    return "Shared content", "📎"


def _media_requested_row(item, command, ctx):
    assignees = command.get("resolved_assignee_ids") or []
    if not assignees:
        for change in command.get("changes") or []:
            if change.get("field") in {"assignee", "owner"}:
                assignees = change.get("value")
                assignees = assignees if isinstance(assignees, list) else [assignees]
                break
    return slack_presentation.TaskRow(
        name=item.title,
        assignee=(", ".join(user_name(user_id) for user_id in assignees) or "Unassigned")
        if config.can_read_field(ctx, "assignee") else None,
        due_date=item.due_date,
        show_due=config.can_read_field(ctx, "due_date"),
        priority=(item.priority or "No priority") if config.can_read_field(ctx, "priority") else None,
        status="Pending" if (config.can_read_field(ctx, "status")
                             or config.can_read_field(ctx, "completed")) else None,
    )


def _execute_media_actions(resolved, ctx, memory_key):
    """Run resolved media actions through the existing handlers and classify presentation."""
    results = []
    schema = slack_tools.get_list_schema(ctx.list_id)
    for item, command in resolved:
        try:
            response = _dispatch(command, ctx, memory_key)
            normalized = str(response or "").casefold()
            if command["intent"] == "create" and (
                    "different fields" in normalized or "existing task differs" in normalized):
                category = "different"
            elif command["intent"] == "create" and "already exists" in normalized:
                category = "duplicate"
            elif command["intent"] == "create" and (
                    "task created" in normalized or "tasks created" in normalized):
                category = "created"
            elif command["intent"] == "create" and "possible duplicate" in normalized:
                category = "clarification"
            elif command["intent"] in {"update", "complete", "reopen"} and "not all changes verified" not in normalized:
                category = "updated"
            else:
                category = "failed"
            results.append({
                "item": item, "command": command, "category": category,
                "response": response, "existing": None,
            })
        except PermissionError as exc:
            results.append({"item": item, "command": command, "category": "permission",
                            "response": str(exc), "existing": None})
        except ValueError as exc:
            results.append({"item": item, "command": command, "category": "clarification",
                            "response": str(exc), "existing": None})
    # One final snapshot supplies verified presentation data for every result;
    # do not perform an additional Slack List read per extracted item.
    current_items = slack_tools.list_action_items(ctx, ctx.list_id) if results else []
    by_title = {}
    for candidate in current_items:
        by_title.setdefault(
            normalize_task_name(slack_tools.extract_item_name(candidate, schema)), []).append(candidate)
    for result in results:
        matches = by_title.get(normalize_task_name(result["item"].title), [])
        result["existing"] = matches[0] if len(matches) == 1 else None
    return results, schema


def _render_media_results(items, results, unresolved, source_types, schema, ctx, warnings=()):
    label, emoji = _media_source_label(source_types)
    lines = [f"*{emoji} {label} processed · {len(items)} action item"
             f"{'s' if len(items) != 1 else ''} extracted*"]
    grouped = {category: [] for category in (
        "created", "updated", "duplicate", "different", "clarification", "permission", "failed")}
    for result in results:
        grouped[result["category"]].append(result)
    headings = {
        "created": "Created", "updated": "Updated", "duplicate": "Already exists",
        "different": "Existing task differs", "clarification": "Needs clarification",
        "permission": "Permission denied", "failed": "Failed",
    }
    for category, group in grouped.items():
        if not group:
            continue
        lines.extend(("", f"*{headings[category]} · {len(group)}*"))
        for result in group:
            item = result["item"]
            requested = _media_requested_row(item, result["command"], ctx)
            if category == "different" and result.get("existing"):
                existing = _task_rows([result["existing"]], schema, ctx)[0]
                owner = existing.assignee or requested.assignee
                identity = f"• *{slack_presentation.text(item.title)}*"
                if owner:
                    identity += f" · {slack_presentation.text(owner)}"
                requested_values = " · ".join((
                    slack_presentation.text(slack_presentation.compact_priority(requested.priority)),
                    slack_presentation.text(slack_presentation.compact_date(
                        requested.due_date, current_date())),
                ))
                existing_values = " · ".join((
                    slack_presentation.text(slack_presentation.compact_priority(existing.priority)),
                    slack_presentation.text(slack_presentation.compact_date(
                        existing.due_date, current_date())),
                ))
                lines.extend((identity, f"  Requested: {requested_values}",
                              f"  Existing: {existing_values}"))
                continue
            display = (_task_rows([result["existing"]], schema, ctx)[0]
                       if result.get("existing") else requested)
            lines.append(slack_presentation.task_line(
                display, bullet=True, show_status=False, today=current_date()))
            if category in {"permission", "clarification", "failed"} and result.get("response"):
                detail = re.sub(r"\s+", " ", str(result["response"])).strip()
                lines.append(f"  {slack_presentation.text(detail[:300])}")
    if unresolved:
        lines.extend(("", f"*Needs clarification · {len(unresolved)}*"))
        for item, message in unresolved:
            if item.assignee:
                lines.append(f"• *{slack_presentation.text(item.title)}* · assignee unclear")
                lines.append(
                    f"  Member clarification required: I couldn't confidently match "
                    f"\"{slack_presentation.text(item.assignee)}\". "
                    "Please use an @mention or display name.")
            else:
                lines.append(f"• *{slack_presentation.text(item.title)}* · clarification required")
                lines.append(f"  Clarification required: {slack_presentation.text(message)}")
    successful = sum(result["category"] in {"created", "updated"} for result in results)
    created = sum(result["category"] == "created" for result in results)
    updated = sum(result["category"] == "updated" for result in results)
    summary = []
    for category, summary_label in (
            ("created", "created"), ("updated", "updated"), ("duplicate", "existing"),
            ("different", "conflict"), ("clarification", "clarification"),
            ("permission", "denied"), ("failed", "failed")):
        if grouped[category]:
            summary.append(f"{len(grouped[category])} {summary_label}")
    if unresolved:
        summary.append(f"{len(unresolved)} clarification")
    if summary:
        lines.extend(("", "*Summary* · " + " · ".join(summary)))
    if created:
        lines.append(f"{created} task{'s' if created != 1 else ''} created successfully and verified.")
    if updated:
        lines.append(f"{updated} task{'s' if updated != 1 else ''} updated successfully and verified.")
    if any(result["category"] == "different" for result in results):
        lines.append("Task already exists with different fields; No fields were changed.")
    if unresolved:
        if successful:
            lines.append("The resolved tasks were processed; tasks needing clarification were not created.")
        else:
            lines.append("No tasks were created.")
        lines.append("Please provide an @mention or exact display name for the task above.")
    if warnings:
        lines.extend(("", "*Content warnings*", *(f"• {warning}" for warning in warnings)))
    return "\n".join(lines)


def _media_no_actions(source_types, warnings=()):
    label, emoji = _media_source_label(source_types)
    message = (f"*{emoji} {label} processed*\n\n"
               "I couldn't identify any clear action items from the transcript.")
    if warnings:
        message += "\n\n*Content warnings*\n" + "\n".join(f"• {warning}" for warning in warnings)
    return message


def _media_failure_message(exc, source_types):
    label, emoji = _media_source_label(source_types)
    message = str(exc)
    normalized = message.casefold()
    if "no usable audio track" in normalized and "video" in set(source_types or []):
        return "*🎥 Video received, but no audio track was found.*"
    if "empty transcript" in normalized or "no usable text" in normalized:
        return (f"*❌ {emoji} {label} transcription failed*\n\n"
                "No usable speech or transcript text was detected. Please upload a clearer file and try again.")
    if ("not configured" in normalized or "requires openai_api_key" in normalized
            or "requires media_transcription_command" in normalized
            or "whisper transcription backend is not installed" in normalized):
        return slack_presentation.failure(
            f"{label} transcription is not available in this workspace.",
            next_step="Ask a workspace administrator to configure the speech-to-text provider.")
    if "ffmpeg" in normalized or "ffprobe" in normalized:
        return slack_presentation.failure(
            f"{label} processing is unavailable because a required media tool is missing.",
            next_step="Ask a workspace administrator to verify the media runtime.")
    if "timed out" in normalized or "temporarily unavailable" in normalized or "rate limited" in normalized:
        return slack_presentation.failure(
            f"{label} transcription is temporarily unavailable.",
            next_step="Please try the upload again in a few minutes.")
    if "unsupported content type" in normalized:
        return slack_presentation.failure("That file type is not supported.",
            next_step="Upload MP3, WAV, M4A, OGG, WEBM, MP4, or MOV media.")
    if isinstance(exc, action_item_extraction.ExtractionError):
        return ("*❌ Action-item extraction failed*\n\n"
                "The AI extraction service could not process the transcript. "
                "Your Slack List data was not changed.")
    return f"*❌ {emoji} {label} processing failed*\n\n{slack_presentation.text(message)}"


def _media_processing_message(files=(), attachments=()):
    kinds = {content_ingestion.file_kind(value)
             for value in [*(files or []), *(attachments or [])]}
    kinds.discard("unsupported")
    label, emoji = _media_source_label(kinds)
    return f"*{emoji} Processing {label.casefold()}…*"


def process_shared_content(text, files, attachments, user_id, channel_id,
                           thread_ts=None, msg_ts=None, team_id=None):
    """Normalize media commands or preserve explicit action-item extraction."""
    started_at = time.monotonic()
    ctx = context(user_id, channel_id, thread_ts, msg_ts, team_id)
    if not ctx.list_id:
        return slack_presentation.failure(
            "This channel is not connected to an Action Items list.",
            next_step="Ask a workspace administrator to map this channel to a Slack List.")
    extraction_mode = bool(
        content_ingestion.extraction_requested(text) or content_ingestion.preview_requested(text)
        or re.search(r"\b(?:apply|extract|create|capture|identify|derive|update)\b.*"
                     r"\b(?:from|in)\s+(?:this|the)\s+"
                     r"(?:audio|video|recording|transcript|attachment|file)\b",
                     str(text or ""), re.I | re.S))
    if extraction_mode and not any(config.has_permission(ctx, intent)
                                   for intent in ("create", "update", "complete", "reopen")):
        return slack_presentation.permission_denied(
            "Your role cannot process action-item changes from shared content.")
    try:
        def build_extraction():
            contents, warnings = content_ingestion.ingest(
                text, files, attachments, BOT_TOKEN, slack_client=slack_tools.client())
            if len(contents) == 1 and not extraction_mode:
                request = content_ingestion.normalized_request(
                    contents[0], requester_id=user_id, channel_id=channel_id,
                    thread_context=thread_ts or msg_ts)
                return {"normalized_request": {
                    "source_type": request.source_type, "raw_input": request.raw_input,
                    "transcript": request.transcript, "normalized_text": request.normalized_text,
                    "requester_id": request.requester_id, "channel_id": request.channel_id,
                    "thread_context": request.thread_context, "source_file": request.source_file},
                    "warnings": warnings, "source_types": [request.source_type]}
            logger.info("action_item_extraction_started sources=%d source_types=%s",
                        len(contents), [content.source_type for content in contents])
            items = action_item_extraction.extract(contents, current_date())
            logger.info("action_items_extracted item_count=%d", len(items))
            return {
                "items": action_item_extraction.serializable(items), "warnings": warnings,
                "source_types": list(dict.fromkeys(content.source_type for content in contents)),
            }
        plan = delivery.stable_plan(
            {"content": "media_ingestion", "mode": "extract" if extraction_mode else "command",
             "files": [str(f.get("id") or f.get("name") or "") for f in files or []]},
            build_extraction)
        if plan.get("normalized_request"):
            request = content_ingestion.NormalizedRequest(**plan["normalized_request"])
            logger.info("normalized_request_ready source=%s actor_id=%s channel_id=%s transcript_chars=%d",
                        request.source_type, user_id, channel_id, len(request.normalized_text))
            return _process(request.normalized_text, user_id, channel_id, thread_ts, msg_ts,
                            team_id, source_type=request.source_type,
                            source_file=request.source_file)
        items = action_item_extraction.deserialize(plan.get("items") or [])
        warnings = plan.get("warnings") or []
        source_types = plan.get("source_types") or [
            content_ingestion.file_kind(value) for value in [*(files or []), *(attachments or [])]
            if content_ingestion.file_kind(value) != "unsupported"]
        if items:
            source_types = list(dict.fromkeys(item.source_type for item in items))
    except (content_ingestion.ContentError, action_item_extraction.ExtractionError) as exc:
        logger.warning("Shared-content processing failed error_type=%s message=%s",
                       type(exc).__name__, str(exc))
        source_types = [content_ingestion.file_kind(value)
                        for value in [*(files or []), *(attachments or [])]
                        if content_ingestion.file_kind(value) != "unsupported"]
        if not source_types and content_ingestion.extraction_requested(text):
            source_types = ["transcript"]
        return _media_failure_message(exc, source_types)
    if not items:
        logger.info("shared_content_processing_completed extracted=0 created=0 skipped=0 "
                    "clarification=0 duration_ms=%d", (time.monotonic() - started_at) * 1000)
        return _media_no_actions(source_types, warnings)
    ambiguous = [(item, item.clarification) for item in items if item.clarification]
    processable = [item for item in items if not item.clarification]
    resolved, unresolved = _resolve_media_actions(processable, ctx)
    unresolved = [*ambiguous, *unresolved]
    if not resolved:
        schema = slack_tools.get_list_schema(ctx.list_id)
        response = _render_media_results(items, [], unresolved, source_types, schema, ctx, warnings)
        logger.info("shared_content_processing_completed extracted=%d created=0 skipped=0 "
                    "clarification=%d duration_ms=%d", len(items), len(unresolved),
                    (time.monotonic() - started_at) * 1000)
        return response
    needs_review = (content_ingestion.preview_requested(text)
                    or any(item.confidence < 0.75 for item, _ in resolved))
    if needs_review:
        schema = slack_tools.get_list_schema(ctx.list_id)
        current_items = slack_tools.list_action_items(ctx, ctx.list_id)
        commands = [command for _, command in resolved]
        command = commands[0] if len(commands) == 1 else validate_command(
            {"intent": "compound", "operations": commands})
        _stage_media_create(ctx, command, [item for item, _ in resolved], schema, current_items)
        note = "Review requested" if content_ingestion.preview_requested(text) else "Low-confidence details require review"
        response = (_media_preview([item for item, _ in resolved])
                    + f"\n_{note}. No tasks were created._\n"
                    "Reply *confirm* within 10 minutes to create these exact items, or *cancel*.")
        if unresolved:
            response += "\n" + _media_member_issues(unresolved)
        if warnings:
            response += "\n" + "\n".join(f"• {warning}" for warning in warnings)
        return response
    results, schema = _execute_media_actions(resolved, ctx, context_keys(ctx)[0])
    response = _render_media_results(items, results, unresolved, source_types, schema, ctx, warnings)
    created = sum(result["category"] == "created" for result in results)
    skipped = sum(result["category"] in {"duplicate", "different"} for result in results)
    logger.info(
        "shared_content_processing_completed extracted=%d created=%d skipped=%d "
        "clarification=%d duration_ms=%d",
        len(items), created, skipped,
        len(unresolved) + sum(result["category"] == "clarification" for result in results),
        (time.monotonic() - started_at) * 1000,
    )
    return response


def process(text, user_id, channel_id, thread_ts=None, msg_ts=None, team_id=None):
    # Serialize the read/resolve/write cycle in this Socket Mode worker.
    with _process_lock:
        _ctx_cleanup()
        request = content_ingestion.normalize_text_request(
            text, requester_id=user_id, channel_id=channel_id,
            thread_context=thread_ts or msg_ts)
        return _process(request.normalized_text, user_id, channel_id, thread_ts, msg_ts,
                        team_id, source_type=request.source_type)


def _process(text, user_id, channel_id, thread_ts=None, msg_ts=None, team_id=None,
             source_type="text", source_file=None):
    ctx = context(user_id, channel_id, thread_ts, msg_ts, team_id)
    try:
        # Pin the original list and parsed command across interrupted event retries.
        original_scope = delivery.stable_plan("scope", lambda: {"list_id": ctx.list_id})
        ctx.list_id = original_scope["list_id"]
        key = context_keys(ctx)[0]
        command_key = delivery.checkpoint_key("command", "interpreted")
        saved = delivery.checkpoint_read(command_key)
        if saved:
            parsed = saved["parsed"]
        else:
            parsed = _interpret(text, ctx)
            delivery.checkpoint_write(command_key, "parsed", {"parsed": parsed})
        intent = parsed.get("intent") or "unknown"
        logger.info(
            "intent_detected source=%s route=command actor_id=%s channel_id=%s intent=%s operation=%s file_id=%s",
            source_type, ctx.user_id, ctx.channel_id, intent, intent, redact(source_file or "none"),
        )
        logger.info("existing_business_logic_started source=%s intent=%s actor_id=%s",
                    source_type, intent, ctx.user_id)
        response = _dispatch(parsed, ctx, key)
        mutation = intent in {"create", "update", "complete", "reopen", "delete", "compound"}
        normalized_response = str(response or "").casefold()
        verification_status = (
            "unverified" if mutation and any(marker in normalized_response for marker in (
                "not verified", "not all changes verified", "outcome unknown"))
            else "completed" if mutation else "not_required")
        logger.info(
            "verification_completed source=%s intent=%s actor_id=%s status=%s",
            source_type, intent, ctx.user_id, verification_status)
        logger.info(
            "request_completed source=%s route=command actor_id=%s intent=%s "
            "operation=%s rbac=allowed outcome=success",
            source_type, ctx.user_id, intent, intent,
        )
        return response
    except PermissionError as exc:
        logger.warning(
            "request_completed source=%s route=command actor_id=%s rbac=denied outcome=failure",
            source_type, ctx.user_id,
        )
        return slack_presentation.permission_denied(str(exc))
    except ValueError as exc:
        logger.info(
            "request_completed source=%s route=command actor_id=%s outcome=clarification",
            source_type, ctx.user_id,
        )
        # Established clarification prompts are already concise and often form
        # part of a scoped conversational selection flow. Preserve them exactly.
        return str(exc)
    except Exception:
        logger.error("Request failed; operation outcome requires verification")
        if delivery.is_active():
            raise
        return slack_presentation.failure(
            "I couldn't verify the outcome of that action-item request.",
            next_step="Check the Action Items list before trying the request again.")


def _clarification_prompt(candidates, *, qualifier=None):
    names = [str(candidate.get("name") or "Action item") for candidate in candidates]
    subject = f"which {qualifier} task" if qualifier else "which task"
    return f"I still need to know {subject} you mean: " + ", ".join(names[:-1]) + (
        f", or {names[-1]}." if len(names) > 1 else f"{names[-1]}.")


def _save_task_clarification(ctx, state, choices, original_intent):
    candidates = [{"item_id": choice.get("item_id"), "name": choice.get("name")}
                  for choice in choices if choice.get("item_id")]
    state = deepcopy(state)
    state["pending_clarification"] = {
        "type": "task_reference", "candidates": candidates,
        "original_intent": original_intent, "created": time.time(),
        "expires_at": time.time() + _CLARIFICATION_TTL,
    }
    _save_state(ctx, state)
    return candidates


def _pending_task_clarification_command(text, state, ctx):
    """Resolve a reply to a pending task clarification without guessing."""
    pending = state.get("pending_clarification") or {}
    if pending.get("type") != "task_reference":
        return None
    if pending.get("expires_at", 0) < time.time():
        state = deepcopy(state)
        state.pop("pending_clarification", None)
        _save_state(ctx, state)
        return None
    candidates = list(pending.get("candidates") or [])
    if not candidates:
        return None
    value = text.strip().casefold().rstrip(".?!")
    if re.fullmatch(
            r"(?:what\s+should\s+(?:i|we)\s+do\s+about\s+(?:it|this|that)(?:\s+task)?|"
            r"why\s+(?:does|is)\s+(?:it|this|that)(?:\s+task)?(?:\s+need\s+attention|\s+risky)?|"
            r"prepare\s+(?:a|the)\s+message(?:\s+for\s+(?:it|this|that))?)", value):
        raise ValueError(_clarification_prompt(candidates))
    selected = []
    reference = parse_reference(text)
    if reference:
        selected_ids = select_ids(reference, [candidate["item_id"] for candidate in candidates])
        selected = [candidate for candidate in candidates
                    if candidate["item_id"] in selected_ids]
    if not selected:
        selected = [candidate for candidate in candidates
                    if value == str(candidate.get("name") or "").strip().casefold()]
    qualifier = None
    if not selected and re.fullmatch(r"(?:the\s+)?overdue(?:\s+one|\s+task)?", value):
        qualifier = "overdue"
        schema = slack_tools.get_list_schema(ctx.list_id)
        displayed = {choice.get("item_id"): choice.get("raw_item")
                     for choice in state.get("displayed_tasks") or []}
        today = current_date()
        for candidate in candidates:
            item = displayed.get(candidate["item_id"])
            if not item:
                continue
            try:
                due = date.fromisoformat(str(slack_tools.extract_due_date(item, schema))[:10])
            except (TypeError, ValueError):
                continue
            if not slack_tools.extract_completed(item, schema) and due < today:
                selected.append(candidate)
    if len(selected) != 1:
        narrowed = selected if selected else candidates
        raise ValueError(_clarification_prompt(narrowed, qualifier=qualifier))
    chosen = selected[0]
    state = deepcopy(state)
    state.pop("pending_clarification", None)
    state.update(last_task_id=chosen["item_id"], last_task_name=chosen.get("name"),
                 last_intent=pending.get("original_intent"))
    _save_state(ctx, state)
    return {
        "intent": "sentinel", "sentinel_mode": "explain",
        "target_ids": [chosen["item_id"]], "target_scope": "single",
        "reference": {"kind": "focus", "positions": [], "count": 0},
        "context_task_id": chosen["item_id"],
    }


def _contextual_task_command(text, state, ctx):
    """Resolve narrow task pronouns only from scoped, structured display state."""
    value = text.strip().casefold().rstrip(".?!")
    explain = re.fullmatch(
        r"(?:why\s+does\s+(?:that|this|it)(?:\s+task)?\s+need\s+attention|"
        r"why\s+is\s+(?:that|this|it)(?:\s+task)?\s+risky|"
        r"what\s+should\s+i\s+do\s+about\s+(?:that|this|it)(?:\s+task)?)", value)
    inspect = re.fullmatch(r"what\s+about\s+(?:that|this|it)(?:\s+task)?", value)
    prepare = re.fullmatch(r"prepare\s+(?:a|the)\s+message\s+for\s+(?:that|this|it)(?:\s+task)?", value)
    if not (explain or inspect or prepare):
        return None
    # A named owner-risk conversation is more specific than a task pronoun.
    if state.get("command_center_owner_id") and not state.get("last_task_id"):
        return None
    task_id = state.get("last_task_id")
    if not task_id:
        choices = state.get("displayed_tasks") or []
        names = [str(choice.get("name") or "Action item") for choice in choices[:3]]
        if len(names) > 1:
            candidates = _save_task_clarification(
                ctx, state, choices[:3], "risk_explanation")
            raise ValueError(_clarification_prompt(candidates))
        if len(choices) == 1:
            task_id = choices[0].get("item_id")
    if not task_id:
        raise ValueError("I don't have one clear recent task in this conversation. Please name the task.")
    trusted_target = {
        "target_ids": [task_id], "target_scope": "single",
        "reference": {"kind": "focus", "positions": [], "count": 0},
        "context_task_id": task_id,
    }
    if prepare:
        return {"intent": "command_center", "command_center_mode": "prepare_message",
                **trusted_target}
    if inspect:
        return {"intent": "inspect", **trusted_target}
    return {"intent": "sentinel", "sentinel_mode": "explain", **trusted_target}


def _interpret(text, ctx):
    pending = _state(ctx)
    plan_followup = text.strip().casefold().rstrip(".?!")
    if pending.get("orchestrator_plans"):
        numbered_remove = re.fullmatch(r"remove\s+step\s+(\d+)", plan_followup)
        if numbered_remove:
            return {"intent": "orchestrator", "orchestrator_mode": "remove_step",
                    "step_number": int(numbered_remove.group(1))}
        if re.fullmatch(r"(?:remove\s+(?:that|the)\s+step|don['’]?t\s+send\s+(?:the\s+)?reminder)",
                        plan_followup):
            return {"intent": "orchestrator", "orchestrator_mode": "remove_step"}
        if re.fullmatch(r"why\s+did\s+you\s+include\s+(?:that|the)\s+(?:reminder|step)",
                        plan_followup):
            return {"intent": "orchestrator", "orchestrator_mode": "explain"}
        if re.fullmatch(r"(?:now\s+)?approve\s+(?:it|that\s+plan)", plan_followup):
            return {"intent": "orchestrator", "orchestrator_mode": "approve"}
    explicit_action = bool(re.fullmatch(
        r"(?:send|dismiss)\s+sentinel\s+alert\s+\d+", text.strip(), re.I))
    explicit_mutation = bool(re.match(
        r"^(?:add|create|assign|reassign|move|update|change|edit|complete|finish|reopen|delete|remove)\b",
        text.strip(), re.I))
    if not explicit_action and not explicit_mutation:
        clarification = _pending_task_clarification_command(text, pending, ctx)
        if clarification:
            return clarification
    if pending.get("visual_context"):
        visual_followup = text.strip().casefold().rstrip(".?!")
        if re.fullmatch(r"(?:visuali[sz]e|graph|chart|plot)\s+(?:it|that|this)", visual_followup):
            kind = pending["visual_context"].get("kind")
            if kind in visual_analytics.KINDS:
                return {"intent": "visual_analytics", "visualization_type": kind,
                        "response_mode": ("dashboard" if kind == "dashboard" else "chart"),
                        "explicit_visual": True}
        requested_type = None
        if re.fullmatch(r"show\s+(?:it|that|this)\s+as\s+(?:a\s+)?pie(?:\s+chart)?", visual_followup):
            requested_type = "pie"
        elif re.fullmatch(r"show\s+(?:it|that|this)\s+as\s+(?:a\s+)?bar(?:\s+chart)?", visual_followup):
            requested_type = "bar"
        elif re.fullmatch(r"put\s+(?:it|that|this)\s+in(?:to)?\s+(?:a\s+)?table", visual_followup):
            requested_type = "table"
        if requested_type:
            kind = pending["visual_context"].get("kind")
            if kind in visual_analytics.KINDS:
                return {"intent": "visual_analytics", "visualization_type": kind,
                        "response_mode": "table" if requested_type == "table" else "chart",
                        "chart_type": requested_type, "explicit_visual": True}
        if re.fullmatch(r"(?:what\s+about\s+priorities|and\s+priorities)", visual_followup):
            return {"intent": "visual_analytics", "visualization_type": "priority",
                    "response_mode": "chart", "explicit_visual": True}
        if re.fullmatch(r"(?:what\s+about\s+deadlines|and\s+deadlines)", visual_followup):
            return {"intent": "visual_analytics", "visualization_type": "deadlines",
                    "response_mode": "chart", "explicit_visual": True}
        if re.fullmatch(r"which\s+one\s+(?:is\s+the\s+biggest\s+concern|needs\s+attention)",
                        visual_followup):
            return {"intent": "sentinel", "sentinel_mode": "risks"}
    member_candidates = pending.get("member_candidates") or []
    member_pending = pending.get("member_pending") or {}
    if member_candidates and member_pending:
        member_ref = parse_reference(text)
        chosen = select_ids(member_ref, [x["id"] for x in member_candidates]) if member_ref else []
        if not chosen:
            exact = [x["id"] for x in member_candidates if text.strip().casefold() in {x["id"].casefold(), x["label"].casefold(), f"<@{x['id']}>".casefold()}]
            chosen = exact if len(exact) == 1 else []
        if len(chosen) == 1:
            parsed = deepcopy(member_pending["command"])
            _set_path(parsed, member_pending["path"], chosen[0])
            pending.pop("member_candidates", None)
            pending.pop("member_pending", None)
            _save_state(ctx, pending)
            return _resolve_command_members(parsed, ctx)
    candidates = pending.get("candidates") or []
    ref = parse_reference(text)
    if candidates:
        ids = None
        if ref:
            ids = select_ids(ref, [slack_tools.extract_item_id(x) for x in candidates])
        else:
            schema = slack_tools.get_list_schema(ctx.list_id)
            named = [x for x in candidates if slack_tools.extract_item_name(x, schema).casefold() == text.strip().casefold()]
            if len(named) == 1:
                ids = [slack_tools.extract_item_id(named[0])]
        if ids:
            parsed = deepcopy(pending["parsed"])
            parsed["target_ids"] = ids
            parsed["target_scope"] = "single" if len(ids) == 1 else "multiple"
            parsed["reference"] = ({"kind": "focus", "positions": [], "count": 0} if len(ids) == 1
                                   else {"kind": "both" if len(ids) == 2 else "focus_set",
                                         "positions": [], "count": 0})
            return parsed
    parsed = _contextual_task_command(text, pending, ctx)
    if parsed is None:
        parsed = _resolve_command_members(validate_command(parse_intent(text)), ctx)
    source = {
        "type": "slack_thread" if ctx.thread_ts else "slack_message",
        "reference": ctx.thread_ts or ctx.msg_ts,
        "evidence": text,
    }
    if parsed.get("intent") == "create" and not parsed.get("_source"):
        parsed["_source"] = source
    elif parsed.get("intent") == "compound":
        for operation in parsed.get("operations") or []:
            if operation.get("intent") == "create" and not operation.get("_source"):
                operation["_source"] = source
    if candidates and parsed["intent"] in {"create", "list", "inspect", "update", "complete", "reopen", "delete"} and not reference_from(parsed):
        pending.pop("candidates", None)
        pending.pop("parsed", None)
        _save_state(ctx, pending)
    return parsed


def _dispatch(parsed, ctx, key):
    intent = parsed["intent"]
    if intent == "compound":
        results = []
        for operation in parsed.get("operations") or []:
            try:
                results.append(_dispatch(operation, ctx, key))
            except (PermissionError, ValueError) as exc:
                results.append(str(exc))
        return "\n\n".join(result for result in results if result)
    if intent == "temporarily_unavailable":
        return ("The AI extraction service is temporarily unavailable.\n\n"
                "Your Slack List data was not changed.")
    if intent == "out_of_scope":
        return OUT_OF_SCOPE
    if intent == "clarify":
        return parsed.get("clarification") or "Please clarify your action-item request."
    if intent == "create":
        duplicate_prompt = maybe_confirm_duplicate_create(parsed, ctx)
        if duplicate_prompt:
            return duplicate_prompt
        tasks = parsed.get("tasks") or []
        return handle_create_multi(tasks, ctx) if tasks else handle_create(parsed, ctx)
    if intent == "list":
        if parsed.get("focus_intelligence"):
            return handle_focus(parsed, ctx, key)
        return handle_list(parsed, ctx, key)
    if intent == "inspect":
        return handle_inspect(parsed, ctx, key)
    if intent == "source":
        return handle_source(parsed, ctx, key)
    if intent == "progress":
        return handle_progress(parsed, ctx, key)
    if intent == "focus":
        return handle_focus(parsed, ctx, key)
    if intent == "weekly_focus":
        return handle_weekly_focus(parsed, ctx, key)
    if intent == "health":
        return handle_health(parsed, ctx, key)
    if intent == "sentinel":
        return handle_sentinel(parsed, ctx, key)
    if intent == "command_center":
        return handle_command_center(parsed, ctx, key)
    if intent == "intelligence_summary":
        return handle_intelligence_summary(parsed, ctx, key)
    if intent == "orchestrator":
        return handle_orchestrator(parsed, ctx, key)
    if intent == "simulation":
        return handle_simulation(parsed, ctx, key)
    if intent == "visual_analytics":
        return handle_visual_analytics(parsed, ctx, key)
    if intent == "plan":
        return handle_plan(parsed, ctx, key)
    if intent == "workload":
        return handle_workload(parsed, ctx)
    if intent == "similar_tasks":
        return handle_similar_tasks(parsed, ctx, key)
    if intent == "standup":
        return handle_standup(parsed, ctx, key)
    if intent == "weekly_summary":
        return handle_weekly_summary(parsed, ctx, key)
    if intent == "apply_proposal":
        return handle_apply_proposal(ctx)
    if intent == "confirm":
        return handle_confirmation(ctx, key)
    if intent == "cancel":
        return handle_cancel(ctx)
    if intent == "history":
        return handle_history(parsed, ctx, key)
    if intent == "dependencies":
        return handle_dependencies(parsed, ctx, key)
    if intent == "members":
        return handle_members(parsed, ctx)
    if intent in {"update", "complete", "reopen", "delete"}:
        return handle_mutation(parsed, ctx, key)
    return None


_MENTION_RE = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]+)?>", re.I)


def _visual_png_bytes(visual):
    """Decode and validate one generated PNG before it reaches Slack."""
    content = base64.b64decode(visual["content_base64"], validate=True)
    if len(content) < 8 or not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Generated visualization is not a valid PNG file.")
    return content


def _slack_upload_diagnostics(exc):
    """Return non-sensitive Slack upload diagnostics without tokens or payloads."""
    details = {"error": type(exc).__name__, "needed": None, "provided": None}
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            details.update(
                error=response.get("error") or "slack_api_error",
                needed=response.get("needed"),
                provided=response.get("provided"),
            )
        except Exception:
            pass
    return details


def _upload_visual(channel, visual, response_text, thread_ts=None):
    """Upload a generated PNG as the primary Slack response.

    Slack's v2 uploader reads the path synchronously, so the request-scoped
    temporary directory remains alive until upload completion and is then
    cleaned automatically.
    """
    content = _visual_png_bytes(visual)
    safe_name = Path(str(visual.get("filename") or "task-visualization.png")).name
    if not safe_name.casefold().endswith(".png"):
        safe_name += ".png"
    with tempfile.TemporaryDirectory(prefix="slack-task-visual-") as directory:
        path = Path(directory) / safe_name
        path.write_bytes(content)
        exists = path.is_file()
        readable = exists and os.access(path, os.R_OK)
        size = path.stat().st_size if exists else 0
        logger.info("visual_file path=%s exists=%s readable=%s size=%d format=png",
                    path, str(exists).lower(), str(readable).lower(), size)
        if not exists or not readable or size <= 0 or size != len(content):
            raise OSError("Generated visualization file could not be prepared.")
        logger.info("visual_generation success=true file_created=true bytes=%d", len(content))
        kwargs = {
            "channel": channel,
            "file": str(path),
            "filename": safe_name,
            "title": visual.get("title") or safe_name,
            "alt_txt": visual.get("title") or "Task analytics visualization",
            "initial_comment": _validated_slack_text(
                response_text, call_site="main._upload_visual->files_upload_v2"),
        }
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        logger.info("visual_upload started=true channel_id=%s filename=%s", channel, safe_name)
        response = app.client.files_upload_v2(**kwargs)
        if response is not None and hasattr(response, "get") and response.get("ok") is False:
            raise RuntimeError(response.get("error") or "Slack rejected the visualization upload.")
        uploaded_file = response.get("file") if response is not None and hasattr(response, "get") else None
        if not uploaded_file and response is not None and hasattr(response, "get"):
            files = response.get("files") or []
            uploaded_file = files[0] if files else None
        file_id = uploaded_file.get("id") if isinstance(uploaded_file, dict) else None
        logger.info("visual_upload success=true channel_id=%s filename=%s file_id=%s",
                    channel, safe_name, file_id or "unavailable")
        return response


def send_and_record(channel, text, thread_ts=None, user_id=None, msg_ts=None, team_id=None, event_key=None):
    visual = text.get("visual") if isinstance(text, dict) else None
    response_text = text.get("text", "") if isinstance(text, dict) else text
    fallback_text = text.get("fallback_text", response_text) if isinstance(text, dict) else response_text
    metadata = {"event_type": "action_item_response", "event_payload": {"request_id": event_key}} if event_key else None
    resp = None
    if visual:
        try:
            resp = _upload_visual(channel, visual, response_text, thread_ts)
        except Exception as exc:
            details = _slack_upload_diagnostics(exc)
            logger.exception(
                "visual_upload success=false channel_id=%s error_type=%s error_code=%s "
                "needed_scope=%s provided_scopes=%s fallback=text",
                channel, type(exc).__name__, details["error"], details["needed"],
                details["provided"],
            )
            fallback = ("I couldn't upload the visualization right now, so here's "
                        "the task summary instead.\n\n" + str(fallback_text or "")).strip()
            resp = post(channel, fallback, thread_ts, metadata=metadata)
    else:
        resp = post(channel, response_text, thread_ts, metadata=metadata)
    bot_ts = None
    if resp:
        if isinstance(resp, dict):
            bot_ts = resp.get("ts")
        elif hasattr(resp, "get"):
            bot_ts = resp.get("ts")
        elif hasattr(resp, "data") and isinstance(resp.data, dict):
            bot_ts = resp.data.get("ts")
    if bot_ts:
        record_bot_response(channel_id=channel, bot_ts=bot_ts, thread_ts=thread_ts, msg_ts=msg_ts, user_id=user_id, team_id=team_id)
    logger.info("response_sent channel_id=%s thread_present=%s response_chars=%d",
                channel, bool(thread_ts), len(str(response_text or "")))
    return resp


def update_and_record(channel, message_ts, text, thread_ts=None, user_id=None,
                      msg_ts=None, team_id=None):
    """Replace a processing notice with the final response and publish its context."""
    resp = app.client.chat_update(
        channel=channel, ts=message_ts,
        text=_validated_slack_text(text, call_site="main.update_and_record->chat_update"))
    record_bot_response(channel_id=channel, bot_ts=message_ts, thread_ts=thread_ts,
                        msg_ts=msg_ts, user_id=user_id, team_id=team_id)
    logger.info("response_sent channel_id=%s thread_present=%s response_chars=%d mode=update",
                channel, bool(thread_ts), len(str(text or "")))
    return resp


def _strip_bot_mention(text, bolt_context):
    bot_id = (bolt_context or {}).get("bot_user_id")
    if not bot_id:
        return text
    return re.sub(r"<@" + re.escape(bot_id) + r"(?:\|[^>]+)?>", "", text).strip()


def request_key(body, event, text):
    msg_id = event.get("client_msg_id")
    if msg_id: return f"{body.get('team_id') or event.get('team')}:{msg_id}"
    eid = body.get("event_id") or event.get("event_ts") or event.get("ts")
    return f"{body.get('team_id') or event.get('team')}:" + str(eid or f"{event.get('channel')}:{event.get('user')}:{event.get('ts')}:{text}")


def _deliver(key, text, user, channel, thread_ts=None, msg_ts=None, team_id=None,
             files=None, attachments=None):
    def recover_post():
        cursor = None
        while True:
            kwargs = {"channel": channel, "limit": 100, "include_all_metadata": True}
            if cursor:
                kwargs["cursor"] = cursor
            if thread_ts:
                response = app.client.conversations_replies(ts=thread_ts, **kwargs)
            else:
                if msg_ts:
                    kwargs["oldest"] = msg_ts
                response = app.client.conversations_history(**kwargs)
            data = slack_tools.checked(response)
            for message in data.get("messages", []):
                metadata = message.get("metadata") or {}
                if (metadata.get("event_payload") or {}).get("request_id") == key:
                    record_bot_response(channel, message["ts"], thread_ts, msg_ts, user, team_id)
                    return message["ts"]
            cursor = (data.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                return None

    request_route = content_ingestion.classify_request(
        text, files or [], attachments or [])
    ingest_route = request_route.is_shared_content
    logger.info("request_received actor_id=%s channel_id=%s thread_present=%s",
                user, channel, bool(thread_ts))
    logger.info("input_type_detected type=%s route=%s files=%d attachments=%d",
                request_route.source, request_route.route,
                len(files or []), len(attachments or []))
    logger.info(
        "request_routed source=%s route=%s channel=%s thread_present=%s "
        "files=%d attachments=%d text_chars=%d",
        request_route.source, request_route.route, channel, bool(thread_ts),
        len(files or []), len(attachments or []), len(str(text or "")),
    )

    def run_request():
        if ingest_route:
            status_key = delivery.checkpoint_key("shared_content_status", "processing")
            if not delivery.checkpoint_read(status_key):
                # Persist before posting so a Socket Mode retry never emits a
                # stream of duplicate processing notices.
                delivery.checkpoint_write(status_key, "posting", {})
                try:
                    posted = post(channel, _media_processing_message(files, attachments), thread_ts)
                    delivery.checkpoint_write(
                        status_key, "posted", {"ts": (posted or {}).get("ts")})
                except Exception as exc:
                    logger.warning("shared_content_status_failed error_type=%s", type(exc).__name__)
        return (process_shared_content(text, files or [], attachments or [], user, channel,
                                       thread_ts, msg_ts, team_id)
                if ingest_route
                else process(text, user, channel, thread_ts, msg_ts, team_id))

    def send_response(response):
        if ingest_route:
            status_key = delivery.checkpoint_key("shared_content_status", "processing")
            status = delivery.checkpoint_read(status_key) or {}
            if status.get("ts"):
                return update_and_record(channel, status["ts"], response, thread_ts,
                                         user, msg_ts, team_id)
        return send_and_record(channel, response, thread_ts, user, msg_ts, team_id,
                               event_key=key)

    # Serialize processing AND publication so response aliases describe the actual response.
    with _process_lock:
        delivery.execute_event(
            _db, key,
            run_request,
            send_response,
            recover_post,
            _publish_context,
        )


def add_command(ack, body):
    ack()
    user, channel = body.get("user_id"), body.get("channel_id")
    text = body.get("text", "").strip()
    if not text:
        post(channel, "Usage: `/add <action item>`")
        return
    key = "slash:" + str(body.get("team_id")) + ":" + str(body.get("trigger_id") or f"{channel}:{user}:{text}")
    # /add supplies creation intent, while preserving existing explicit create phrases.
    if not re.match(r"(?:add|create|assign)\b", text, re.I):
        text = "add " + text
    _deliver(key, text, user, channel, team_id=body.get("team_id"))


def app_mention(body, event, context=None):
    if event.get("bot_id"):
        return
    text = _strip_bot_mention(event.get("text", ""), context)
    _deliver("event:" + request_key(body, event, text), text, event.get("user"), event.get("channel"),
             event.get("thread_ts"), event.get("ts"), body.get("team_id") or event.get("team"),
             files=event.get("files"), attachments=event.get("attachments"))


def message_handler(body, event, context=None):
    if event.get("bot_id") or not event.get("user"):
        return
    channel, thread_ts = str(event.get("channel", "")), event.get("thread_ts")
    is_dm = channel.startswith("D")
    text = event.get("text", "")
    bot_id = (context or {}).get("bot_user_id")
    if not is_dm and bot_id and any(m.group(1) == bot_id for m in _MENTION_RE.finditer(text)):
        return
    if not is_dm and not thread_ts:
        return
    text = _strip_bot_mention(text, context)
    _deliver("event:" + request_key(body, event, text), text, event.get("user"), channel,
             thread_ts, event.get("ts"), body.get("team_id") or event.get("team"),
             files=event.get("files"), attachments=event.get("attachments"))


def create_app(client=None):
    global app
    if client is None and (not BOT_TOKEN or not APP_TOKEN):
        raise RuntimeError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required")
    app = App(client=client) if client else App(token=BOT_TOKEN)
    slack_tools.configure(app.client)
    app.command("/add")(add_command)
    app.event("app_mention")(app_mention)
    app.event({"type": "message", "subtype": None})(message_handler)
    app.event({"type": "message", "subtype": "file_share"})(message_handler)
    return app


if __name__ == "__main__":
    logger.info("Slack List Assistant starting; Socket Mode enabled")
    reminder_scheduler = None
    try:
        socket_app = create_app()
        reminder_scheduler = create_reminder_scheduler()
        socket_handler = SocketModeHandler(socket_app, APP_TOKEN)
        reminder_scheduler.start()
        socket_handler.start()
    except KeyboardInterrupt:
        logger.info("Slack List Assistant shutdown requested")
    finally:
        if reminder_scheduler is not None:
            reminder_scheduler.stop()
        transcription.shutdown_transcription_provider()
