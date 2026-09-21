import json
import logging
import os
import re
import sqlite3
import threading
import time
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import config
import slack_tools
import mutations
import delivery
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
    value = str(text or "")
    value = re.sub(r"\\\*\\\*(.+?)\\\*\\\*", r"*\1*", value)
    return re.sub(r"\*\*(.+?)\*\*", r"*\1*", value)


def post(channel, text, thread_ts=None, metadata=None):
    kwargs = {"channel": channel, "text": slack_mrkdwn(text)}
    if metadata:
        kwargs["metadata"] = metadata
    if thread_ts: kwargs["thread_ts"] = thread_ts
    return app.client.chat_postMessage(**kwargs)


def user_name(uid):
    return slack_tools.user_display_name(uid) or "Unknown user"


def fmt_task(item, schema, index, ctx=None):
    readable = lambda field: ctx is None or config.can_read_field(ctx, field)
    name = slack_tools.extract_item_name(item, schema) if readable("name") else "Restricted task"
    lines = [f"{index}. *{name or 'Unnamed task'}*"]
    assignee = slack_tools.extract_assignee(item, schema) if readable("assignee") else None
    if assignee: lines.append(f"   • Assignee: {assignee}")
    due = slack_tools.extract_due_date(item, schema) if readable("due_date") else None
    if due: lines.append(f"   • Due: {due}")
    priority = slack_tools.extract_priority(item, schema) if readable("priority") else None
    if priority: lines.append(f"   • Priority: {priority}")
    if readable("status") or readable("completed"):
        lines.append(f"   • Status: {('Completed' if slack_tools.extract_completed(item, schema) else 'Pending')}")
    return "\n".join(lines)


def format_items(items, schema, title="Action Items", ctx=None):
    if not items: return f"*{title}*\n\nNo action items found."
    return "*" + title + "*\n\n" + "\n".join(fmt_task(i, schema, n, ctx) for n, i in enumerate(items, 1))


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


def _resolve_command_members(parsed, ctx, path=(), root=None):
    """Resolve member text once and retain Slack IDs in the trusted command."""
    root = parsed if root is None else root
    parsed["actor_id"] = ctx.user_id
    if parsed.get("intent") == "compound":
        for index, operation in enumerate(parsed.get("operations") or []):
            _resolve_command_members(operation, ctx, path + ("operations", index), root)
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
            matches = slack_tools.find_user_candidates(value)
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
            matches = slack_tools.find_user_candidates(value)
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
            matches = slack_tools.find_user_candidates(value)
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
        _resolve_command_members(task, ctx, path + ("tasks", index), root)
    return parsed


def _filter_items(items, parsed, schema, assignee_ids=(), default_pending=False):
    """Apply the same structured filters to reads and bulk target resolution."""
    temporal = parsed.get("temporal_filter") or {}
    temporal_field = temporal.get("field")
    if temporal_field == "completed_at":
        raise ValueError(
            "Slack List exposes whether a task is completed, but not a reliable completion timestamp, "
            "so I can't determine which tasks were completed on that date.")

    def item_date(item, field):
        if field == "due_date":
            raw = slack_tools.extract_due_date(item, schema)
        elif field == "created_at":
            raw = item.get("date_created") or item.get("created_timestamp")
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
    completed = parsed.get("completed")
    status = config.normalize_status(parsed.get("status"))
    if status in {"open", "completed"}:
        completed = status == "completed"
    elif (parsed.get("all_tasks") or parsed.get("all")):
        completed = None
    elif completed is None and default_pending:
        completed = False

    today = current_date()
    week_end = today + timedelta(days=6)
    try:
        date_from = date.fromisoformat(parsed["date_from"]) if parsed.get("date_from") else None
        date_to = date.fromisoformat(parsed["date_to"]) if parsed.get("date_to") else None
    except (TypeError, ValueError):
        raise ValueError("I couldn't understand the requested date range.")
    if date_from and date_to and date_from > date_to:
        raise ValueError("The start of the requested date range is after its end.")
    priority = config.normalize_priority(parsed.get("priority"))
    query = (parsed.get("query") or "").strip().casefold()
    assignee_ids = set(assignee_ids)
    assignee_condition = parsed.get("assignee_condition")
    actor_id = parsed.get("actor_id")
    if assignee_condition in {"self", "other"} and not actor_id:
        raise ValueError("I couldn't determine the requesting Slack user for that assignee filter.")
    filtered = []
    for item in items:
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
        if query and query not in slack_tools.extract_item_name(item, schema).casefold():
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
        return slack_tools.extract_assignee(item, schema) or "Unassigned"
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
    return f"*{title} — Grouped by {parsed['group_by'].replace('_', ' ').title()}*\n\n" + ("\n\n".join(lines) or "No matching action items.")


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
    names = "\n".join(f"{i}. {slack_tools.extract_item_name(x, schema)} (ID: {slack_tools.extract_item_id(x)})" for i, x in enumerate(matches, 1))
    raise ValueError(f"Which {name} task do you mean?\n" + names)


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
        assignee_label = ", ".join(filter(None, (slack_tools.user_display_name(uid) for uid in assignee_ids)))
    completed = parsed.get("completed")
    normalized_status = config.normalize_status(parsed.get("status"))
    if normalized_status in {"open", "completed"}:
        completed = normalized_status == "completed"
    elif parsed.get("all_tasks") or parsed.get("all"):
        completed = None
    elif completed is None:
        completed = False

    if sort_by == "due_date" and limit == 1:
        if is_self:
            title = "My Next Task"
        elif is_named and assignee_label:
            title = f"{assignee_label}'s Next Task"
        else:
            title = "Next Task by Deadline"
    elif parsed.get("due_today"):
        if is_self:
            title = "My Tasks Due Today"
        elif is_named and assignee_label:
            title = f"{assignee_label}'s Tasks Due Today"
        else:
            title = "Tasks Due Today"
    elif parsed.get("overdue"):
        title = "Overdue Action Items"
    elif completed is False:
        title = "My Pending Tasks" if is_self else "Pending Action Items"
    elif completed is True:
        title = "Completed Action Items"
    elif is_self:
        title = "My Action Items"
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
        me = slack_tools.user_display_name(ctx.user_id) or "you"
        return (f"*{title}*\n\nNo tasks due today are assigned to {me}.\n"
                f"_Use 'show all tasks due today' to see everyone's tasks._")

    if parsed.get("group_by"):
        return _format_grouped(filtered, parsed, schema, title, ctx)
    if parsed.get("aggregate") == "count" or parsed.get("count_only"):
        return f"{len(filtered)} matching action item(s).\n\n" + format_items(filtered, schema, title, ctx)
    return format_items(filtered, schema, title, ctx)





def handle_create(parsed, ctx):
    if not config.has_permission(ctx, "create"): raise PermissionError("You do not have permission to create action items.")
    name = (parsed.get("task_name") or "").strip()

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
        return "**Missing information**\nPlease specify the action item name."

    # Assignee is optional — tasks can be created unassigned
    assignee = []
    if assignee_raw:
        for value in assignee_raw:
            user_id = slack_tools.find_user_id(value)
            if not user_id:
                return "**Member not found**\nI couldn't find that Slack member. Please mention a valid workspace member."
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
            return "*Task already exists with different fields*\n" + format_items(exact, schema, ctx=ctx) + "\nPlease request an update explicitly; no changes were made."
        return "*Task already exists*\n" + format_items(exact, schema, ctx=ctx) + "\nNo changes were made."
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
    store_view(context_keys(ctx)[0], [item], schema, ctx)
    task_name_out = slack_tools.extract_item_name(item, schema) if config.can_read_field(ctx, "name") else "Restricted task"
    assignee_out = (slack_tools.extract_assignee(item, schema) or "Unassigned") if config.can_read_field(ctx, "assignee") else None
    msg = "*Action item created successfully!*\n\n"
    msg += f"\U0001f4cc *Task:* {task_name_out}\n"
    if assignee_out is not None: msg += f"\U0001f464 *Assignee:* {assignee_out}\n"
    if priority and config.can_read_field(ctx, "priority"): msg += f"\U0001f534 *Priority:* {priority}\n"
    if due_date and config.can_read_field(ctx, "due_date"): msg += f"\U0001f4c5 *Due:* {due_date}\n"
    if config.can_read_field(ctx, "status") or config.can_read_field(ctx, "completed"):
        msg += "\u23f3 *Status:* Pending"
    return msg



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
                if "created successfully" in result or "Task already exists" in result:
                    successes.append(result)
                else:
                    failures.append(f"{i}. *{task_name}* — {result}")
                if "Action item created successfully" in result:
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
    if successes:
        parts.append("\n\n".join(successes))
    if failures:
        failure_block = "*The following tasks could not be created:*\n" + "\n".join(failures)
        parts.append(failure_block)

    if created_items:
        parts.append(format_items(created_items, schema, "Created tasks", ctx))
    return "\n\n".join(parts) if parts else "No tasks were processed."



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
    item_ids = list(mutations.authorize_collection(item_ids, items, intent, changes, ctx, schema))
    # The mutation boundary consumes exact IDs only; it never performs name matching.
    results = mutations.execute_collection(item_ids, intent, changes, ctx, schema)
    names_by_id = {slack_tools.extract_item_id(item): slack_tools.extract_item_name(item, schema) for item in items}
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
    all_verified = all(result.verified for result in results)
    heading = f"Action item{'s' if len(item_ids) > 1 else ''} {verb}" if all_verified else "Action results — not all changes verified"
    lines = []
    for item_id, result in zip(item_ids, results):
        name = names_by_id.get(item_id, item_id)
        if not result.verified:
            lines.append(f"• *{name}*: " + "; ".join(result.problems))
        elif intent == "delete":
            lines.append(f"• *{name}* — deletion verified")
        else:
            # Bullets avoid introducing a second, conflicting set of displayed positions.
            lines.append(re.sub(r"^1\. ", "• ", fmt_task(result.item, schema, 1, ctx)))
    return f"*{heading}*\n\n" + "\n\n".join(lines)


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
    return "\n\n".join(re.sub(r"^1\. ", "• ", fmt_task(x, schema, 1, ctx)) for x in targets)


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
        f"• {member['name']} (<@{member['id']}>) — {config.get_user_role(member['id'], ctx.team_id)}"
        for member in members)


_process_lock = threading.RLock()


def process(text, user_id, channel_id, thread_ts=None, msg_ts=None, team_id=None):
    # Serialize the read/resolve/write cycle in this Socket Mode worker.
    with _process_lock:
        _ctx_cleanup()
        return _process(text, user_id, channel_id, thread_ts, msg_ts, team_id)


def _process(text, user_id, channel_id, thread_ts=None, msg_ts=None, team_id=None):
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
        return _dispatch(parsed, ctx, key)
    except PermissionError as exc:
        return f"Permission denied: {exc}"
    except ValueError as exc:
        return str(exc)
    except Exception:
        logger.error("Request failed; operation outcome requires verification")
        if delivery.is_active():
            raise
        return "I couldn't complete that action-item request. Its outcome could not be verified."


def _interpret(text, ctx):
    pending = _state(ctx)
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
    parsed = _resolve_command_members(validate_command(parse_intent(text)), ctx)
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
        return "The AI model is temporarily unavailable. Please try again in a few moments."
    if intent == "out_of_scope":
        return OUT_OF_SCOPE
    if intent == "clarify":
        return parsed.get("clarification") or "Please clarify your action-item request."
    if intent == "create":
        tasks = parsed.get("tasks") or []
        return handle_create_multi(tasks, ctx) if tasks else handle_create(parsed, ctx)
    if intent == "list":
        return handle_list(parsed, ctx, key)
    if intent == "inspect":
        return handle_inspect(parsed, ctx, key)
    if intent == "members":
        return handle_members(parsed, ctx)
    if intent in {"update", "complete", "reopen", "delete"}:
        return handle_mutation(parsed, ctx, key)
    return None


_MENTION_RE = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]+)?>", re.I)


def send_and_record(channel, text, thread_ts=None, user_id=None, msg_ts=None, team_id=None, event_key=None):
    metadata = {"event_type": "action_item_response", "event_payload": {"request_id": event_key}} if event_key else None
    resp = post(channel, text, thread_ts, metadata=metadata)
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


def _deliver(key, text, user, channel, thread_ts=None, msg_ts=None, team_id=None):
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

    # Serialize processing AND publication so response aliases describe the actual response.
    with _process_lock:
        delivery.execute_event(
            _db, key,
            lambda: process(text, user, channel, thread_ts, msg_ts, team_id),
            lambda response: send_and_record(channel, response, thread_ts, user, msg_ts, team_id, event_key=key),
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
             event.get("thread_ts"), event.get("ts"), body.get("team_id") or event.get("team"))


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
             thread_ts, event.get("ts"), body.get("team_id") or event.get("team"))


def create_app(client=None):
    global app
    if client is None and (not BOT_TOKEN or not APP_TOKEN):
        raise RuntimeError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required")
    app = App(client=client) if client else App(token=BOT_TOKEN)
    slack_tools.configure(app.client)
    app.command("/add")(add_command)
    app.event("app_mention")(app_mention)
    app.event({"type": "message", "subtype": None})(message_handler)
    return app


if __name__ == "__main__":
    logger.info("Slack List Assistant starting; Socket Mode enabled")
    SocketModeHandler(create_app(), APP_TOKEN).start()
