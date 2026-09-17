import logging
import os
import sqlite3
import threading
import time
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
from typing import Optional

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import config
import slack_tools
from intent_parser import parse_intent

def current_date():
    return datetime.now(ZoneInfo("Asia/Kathmandu")).date()

def normalize_task_name(name):
    import re
    # Remove punctuation and collapse whitespace
    name = re.sub(r'[^\w\s]', '', name)
    return re.sub(r'\s+', ' ', name).casefold().strip()

load_dotenv()
BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "").strip()
APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "").strip()
if not BOT_TOKEN or not APP_TOKEN:
    raise RuntimeError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("slack_list")
app = App(token=BOT_TOKEN)
slack_tools.configure(app.client)

from slack_bolt.response import BoltResponse

# No global ignore_retries middleware needed; duplicate checking handles retries

DB_PATH = os.getenv("STATE_DB", "slack_list_state.sqlite3")
_db_lock = threading.Lock()
_pending = {}

OUT_OF_SCOPE = "I can only help with action items and Slack Lists."


def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS events (key TEXT PRIMARY KEY, created REAL NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, status TEXT NOT NULL, created REAL NOT NULL)")
    conn.commit(); return conn


def claim_event(key: str) -> bool:
    if not key: return True
    with _db_lock:
        conn = _db()
        try:
            conn.execute("DELETE FROM events WHERE created < ?", (time.time() - 86400,))
            cur = conn.execute("INSERT OR IGNORE INTO events(key,created) VALUES(?,?)", (key, time.time()))
            conn.commit(); return cur.rowcount == 1
        finally: conn.close()


def claim_request(key: str) -> bool:
    with _db_lock:
        conn = _db()
        try:
            cur = conn.execute("INSERT OR IGNORE INTO requests(key,status,created) VALUES(?,?,?)", (key, "processing", time.time()))
            conn.commit(); return cur.rowcount == 1
        finally: conn.close()


def release_request(key: str):
    with _db_lock:
        conn = _db(); conn.execute("UPDATE requests SET status='done' WHERE key=?", (key,)); conn.commit(); conn.close()


def context(user_id, channel_id, thread_ts=None):
    return config.build_context(user_id=user_id, channel_id=channel_id, thread_ts=thread_ts)


def post(channel, text, thread_ts=None):
    kwargs = {"channel": channel, "text": text}
    if thread_ts: kwargs["thread_ts"] = thread_ts
    return app.client.chat_postMessage(**kwargs)


def user_name(uid):
    return slack_tools.user_display_name(uid) or "Unknown user"


def fmt_task(item, schema, index):
    lines = [f"{index}. *{slack_tools.extract_item_name(item, schema) or 'Unnamed task'}*"]
    assignee = slack_tools.extract_assignee(item, schema)
    if assignee: lines.append(f"   • Assignee: {assignee}")
    due = slack_tools.extract_due_date(item, schema)
    if due: lines.append(f"   • Due: {due}")
    priority = slack_tools.extract_priority(item, schema)
    if priority: lines.append(f"   • Priority: {priority}")
    lines.append(f"   • Status: {('Completed' if slack_tools.extract_completed(item, schema) else 'Pending')}")
    return "\n".join(lines)


def format_items(items, schema, title="Action Items"):
    if not items: return f"*{title}*\n\nNo action items found."
    return "*" + title + "*\n\n" + "\n".join(fmt_task(i, schema, n) for n, i in enumerate(items, 1))


def store_view(key, items):
    _pending[key] = {"items": items, "created": time.time()}
    # Keep memory bounded.
    for k, v in list(_pending.items()):
        if time.time() - v["created"] > 1800: _pending.pop(k, None)


def previous_items(key):
    v = _pending.get(key); return v["items"] if v and time.time() - v["created"] <= 1800 else []


def selection_from(parsed, items, schema):
    selection = parsed.get("selection")
    if selection not in {"single", "first", "all", "both", "numbered"} or not items:
        return []
    if selection == "single": return items[:1]
    if selection == "first": return items[:1]
    if selection == "all": return items[:]
    if selection == "both": return items[:2]
    nums = parsed.get("selection_numbers") or []
    if parsed.get("selection_count"):
        nums = list(range(1, int(parsed["selection_count"]) + 1))
    return [items[n - 1] for n in nums if isinstance(n, int) and 1 <= n <= len(items)]


def resolve_targets(parsed, items, schema, memory_key, ctx=None, intent=None):
    """
    Resolve the list item(s) that a mutation (update/delete/complete/reopen) should act on.

    `assignee` in parsed = the task's target user (used to NARROW the search scope).
    `ctx.user_id` = the REQUESTER (used for RBAC — checked by the caller, not here).

    These are different. An admin can delete any user's task; we still use the parsed
    assignee to find the right task when the command specifies one.
    """
    task_name = parsed.get("task_name")

    # Resolve the target assignee for search-narrowing (NOT for RBAC)
    target_assignee_id = None
    if parsed.get("assignee_self") and ctx:
        # "my task X" → task is assigned to the requester
        target_assignee_id = ctx.user_id
    elif parsed.get("assignee"):
        resolved = slack_tools.find_user_id(parsed["assignee"])
        if not resolved:
            raise ValueError(
                f"I couldn't resolve the user {parsed['assignee']!r}. "
                "Please use their Slack @mention or display name."
            )
        target_assignee_id = resolved

    # A follow-up like "both" uses candidates from the preceding ambiguous request.
    pending = _pending.get(memory_key, {})
    candidates = (
        pending.get("candidates")
        if pending and time.time() - pending.get("created", 0) <= 1800
        else None
    )
    if parsed.get("selection") in {"single", "first", "all", "both", "numbered"} and candidates:
        return selection_from(parsed, candidates, schema)

    if parsed.get("selection") in {"single", "first", "all", "both", "numbered"}:
        base = items
        if target_assignee_id:
            base = [x for x in items if slack_tools.extract_assignee_id(x, schema) == target_assignee_id]
        if parsed.get("task_name") == "__LAST__" and not candidates:
            base = previous_items(memory_key) or base
        return selection_from(parsed, base, schema)

    if task_name in {"__LAST__", "that one", "that task", "the one above"}:
        return selection_from({"selection": "single"}, candidates or previous_items(memory_key), schema)
    if not task_name:
        # No task name — only then do we filter by assignee (e.g. "delete all of @Praveen's tasks")
        if target_assignee_id:
            return [x for x in items if slack_tools.extract_assignee_id(x, schema) == target_assignee_id]
        return []

    # Search ALL items by task name first.
    # The assignee is used as a TIEBREAKER only when multiple tasks share the same name.
    # It must NOT pre-filter the search pool — e.g. "update @AasthaA Ollama Cloud Test to P2"
    # means "find task 'Ollama Cloud Test', using @AasthaA to disambiguate if needed".
    logger.info("RESOLVE task_name=%r pool_size=%d target_assignee_id=%r",
                task_name, len(items), target_assignee_id)
    all_matches = slack_tools.find_matches(items, task_name, schema)
    logger.info("FIND_MATCHES returned %d match(es) for %r", len(all_matches), task_name)

    if len(all_matches) == 0:
        return []

    if len(all_matches) == 1:
        return all_matches

    # Multiple matches — try to narrow using the assignee as a tiebreaker
    if target_assignee_id:
        narrowed = [
            m for m in all_matches
            if slack_tools.extract_assignee_id(m, schema) == target_assignee_id
        ]
        if len(narrowed) == 1:
            return narrowed
        if len(narrowed) > 1:
            all_matches = narrowed   # still ambiguous within assignee's tasks

    # Try exact name match to resolve ambiguity
    exact = [
        m for m in all_matches
        if slack_tools.extract_item_name(m, schema).casefold() == task_name.casefold()
    ]
    if len(exact) == 1:
        return exact

    # Still ambiguous — store candidates and ask for clarification
    _pending[memory_key] = {"candidates": all_matches, "created": time.time()}
    names = "\n".join(f"• {slack_tools.extract_item_name(x, schema)}" for x in all_matches[:8])
    raise ValueError(
        f"I found {len(all_matches)} tasks matching {task_name!r}. "
        "Which one did you mean?\n" + names
    )




def handle_list(parsed, ctx, memory_key):
    list_id = ctx.list_id
    if not list_id: raise ValueError("This channel is not mapped to a Slack List.")
    schema = slack_tools.get_list_schema(list_id)
    items = slack_tools.list_action_items(ctx, list_id)

    # ── Resolve assignee filter ───────────────────────────────────────────
    assignee = None
    if parsed.get("assignee_self"):
        assignee = ctx.user_id
    elif parsed.get("assignee"):
        assignee = slack_tools.find_user_id(parsed["assignee"])
        if not assignee: raise ValueError("I couldn't resolve that Slack user.")

    # ── Completion filter ─────────────────────────────────────────────────
    completed = parsed.get("completed")
    if parsed.get("status") == "open": completed = False
    if parsed.get("status") == "completed": completed = True

    today = current_date()
    today_iso = today.isoformat()
    week_end = today + timedelta(days=6)
    query = parsed.get("query") or None

    filtered = []
    for item in items:
        # Completion filter
        if completed is not None and slack_tools.extract_completed(item, schema) != completed:
            continue
        # Assignee filter
        if assignee and slack_tools.extract_assignee_id(item, schema) != assignee:
            continue
        # Priority filter
        if parsed.get("priority") and slack_tools.extract_priority(item, schema) != config.normalize_priority(parsed["priority"]):
            continue
        # Name search filter
        if query and query.casefold() not in slack_tools.extract_item_name(item, schema).casefold():
            continue

        due = slack_tools.extract_due_date(item, schema)  # always YYYY-MM-DD str or None

        # Due-today filter — compare as date objects to avoid string format issues
        if parsed.get("due_today"):
            if not due:
                continue
            try:
                due_obj = date.fromisoformat(due)
            except ValueError:
                continue
            if due_obj != today:
                continue

        # Overdue filter
        if parsed.get("overdue") and (not due or due >= today_iso or slack_tools.extract_completed(item, schema)):
            continue

        # Due-this-week filter
        if parsed.get("due_this_week"):
            if not due:
                continue
            try:
                d = date.fromisoformat(due)
            except ValueError:
                continue
            if not today <= d <= week_end:
                continue

        filtered.append(item)

    # ── Sorting ───────────────────────────────────────────────────────────
    sort_by = parsed.get("sort_by")
    sort_order = parsed.get("sort_order", "asc")
    if sort_by == "due_date":
        # Tasks with no due date go to the end when sorting ascending
        def _due_key(item):
            d = slack_tools.extract_due_date(item, schema)
            if not d:
                return "9999-99-99"
            return d
        filtered.sort(key=_due_key, reverse=(sort_order == "desc"))

    # ── Limit ─────────────────────────────────────────────────────────────
    limit = parsed.get("limit")
    if limit and isinstance(limit, int) and limit > 0:
        filtered = filtered[:limit]
    elif parsed.get("selection") == "first":
        filtered = filtered[:1]
    elif parsed.get("selection") == "both":
        filtered = filtered[:2]
    elif parsed.get("selection_count"):
        filtered = filtered[:int(parsed["selection_count"])]

    # ── Title ─────────────────────────────────────────────────────────────
    is_self = bool(parsed.get("assignee_self"))
    is_named = bool(parsed.get("assignee") and not is_self)
    assignee_label = None
    if is_named:
        assignee_label = slack_tools.user_display_name(assignee) if assignee else parsed.get("assignee")

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

    store_view(memory_key, filtered)

    # Helpful empty message for self-scoped due-today queries
    if not filtered and parsed.get("due_today") and is_self:
        me = slack_tools.user_display_name(ctx.user_id) or "you"
        return (f"*{title}*\n\nNo tasks due today are assigned to {me}.\n"
                f"_Use 'show all tasks due today' to see everyone's tasks._")

    return format_items(filtered, schema, title)





def handle_create(parsed, ctx):
    if not config.has_permission(ctx, "create"): raise PermissionError("You do not have permission to create action items.")
    name = (parsed.get("task_name") or "").strip()

    assignee_raw = parsed.get("assignee")
    if parsed.get("assignee_self"): assignee_raw = ctx.user_id
    priority = parsed.get("priority")
    due_date = parsed.get("due_date")

    if not name or name == "__LAST__":
        return "**Missing information**\nPlease specify the action item name."

    if not assignee_raw:
        return "**Missing information**\nPlease specify who should be assigned this task."

    import re as _re
    if _re.fullmatch(r"<@[UW][A-Z0-9]+(?:\|[^>]+)?>|[UW][A-Z0-9]+", name): raise ValueError("The task title cannot be just a Slack user ID.")

    if due_date:
        try:
            if due_date < current_date().isoformat():
                return "**Invalid due date**\n📅 The due date cannot be in the past.\n\nPlease provide a future date."
        except Exception:
            pass

    assignee = None
    if assignee_raw:
        assignee = slack_tools.find_user_id(assignee_raw)
        if not assignee:
            return "**Member not found**\nI couldn't find that Slack member. Please mention a valid workspace member."

    schema = slack_tools.get_list_schema(ctx.list_id)
    items = slack_tools.list_action_items(ctx, ctx.list_id)
    normalized = normalize_task_name(name)

    # Check for existing task with same name
    existing_match = None
    for item in items:
        if slack_tools.extract_completed(item, schema): continue
        if normalize_task_name(slack_tools.extract_item_name(item, schema)) == normalized:
            existing_match = item
            break

    if existing_match:
        existing_assignee_id = slack_tools.extract_assignee_id(existing_match, schema)
        existing_assignee = slack_tools.extract_assignee(existing_match, schema)
        existing_name = slack_tools.extract_item_name(existing_match, schema)
        item_id = slack_tools.extract_item_id(existing_match)

        # If it's already assigned to the same person, report duplicate
        if existing_assignee_id == assignee:
            return (
                f"*Task already exists*\n"
                f"📌 *Task:* {existing_name}\n"
                f"👤 *Assignee:* {existing_assignee or 'Not set'}\n"
                f"\nThis task is already assigned to this person. No changes were made."
            )

        # Otherwise, update the assignee (and other fields) on the existing task
        changes = [{"field": "assignee", "value": assignee}]
        if priority:
            changes.append({"field": "priority", "value": config.normalize_priority(priority)})
        if due_date:
            changes.append({"field": "due_date", "value": due_date})

        for change in changes:
            slack_tools.update_action_item_field(item_id, change["field"], change["value"], ctx, ctx.list_id)

        refreshed = slack_tools.list_action_items(ctx, ctx.list_id)
        updated_item = next((x for x in refreshed if slack_tools.extract_item_id(x) == item_id), existing_match)
        task_disp = fmt_task(updated_item, schema, 1)
        return f"*Task updated* (already existed — updated instead of creating a duplicate)\n\n{task_disp}"

    # No existing match — create fresh
    item = slack_tools.create_action_item(name, priority, assignee, due_date, ctx, ctx.list_id)
    schema = slack_tools.get_list_schema(ctx.list_id)
    task_name_out = slack_tools.extract_item_name(item, schema)
    assignee_out = slack_tools.extract_assignee(item, schema) or "Unassigned"
    msg = "*Action item created successfully!*\n\n"
    msg += f"📌 *Task:* {task_name_out}\n"
    msg += f"👤 *Assignee:* {assignee_out}\n"
    if priority: msg += f"🔴 *Priority:* {priority}\n"
    if due_date: msg += f"📅 *Due:* {due_date}\n"
    msg += "⏳ *Status:* Pending"
    return msg



def handle_mutation(parsed, ctx, memory_key):
    intent = parsed["intent"]
    items = slack_tools.list_action_items(ctx, ctx.list_id)
    schema = slack_tools.get_list_schema(ctx.list_id)
    logger.info("MUTATION intent=%s task_name=%r assignee=%r list_id=%s item_count=%d",
                intent, parsed.get("task_name"), parsed.get("assignee"), ctx.list_id, len(items))
    for i, item in enumerate(items):
        logger.debug("  ITEM[%d] name=%r assignee_id=%r", i,
                     slack_tools.extract_item_name(item, schema),
                     slack_tools.extract_assignee_id(item, schema))
    # resolve_targets handles assignee resolution internally (for search narrowing only).
    # RBAC is checked below, based on ctx (the requester), NOT on the task's assignee.
    targets = resolve_targets(parsed, items, schema, memory_key, ctx=ctx, intent=intent)
    if not targets:
        logger.warning("RESOLVE_TARGETS returned empty for task_name=%r assignee=%r",
                       parsed.get("task_name"), parsed.get("assignee"))
        raise ValueError(
            "I couldn't identify the action item. "
            "Please specify its name, or say 'first', 'both', or 'all' after a list."
        )
    logger.info("RESOLVED %d target(s): %s", len(targets),
                [slack_tools.extract_item_name(t, schema) for t in targets])

    if intent == "delete" and len(targets) > 1:
        # Explicit multi-delete is accepted; otherwise ambiguity is rejected by resolver.
        pass
    changed = []
    for item in targets:
        item_id = slack_tools.extract_item_id(item)
        name = slack_tools.extract_item_name(item, schema)
        if intent == "delete":
            if not config.has_permission(ctx, "delete"): raise PermissionError("You do not have permission to delete action items.")
            slack_tools.delete_action_item(item_id, ctx, ctx.list_id); changed.append(f"• {name}")
        elif intent == "complete":
            if not config.has_permission(ctx, "complete"): raise PermissionError("You do not have permission to complete action items.")
            res = slack_tools.complete_action_item(item_id, ctx, ctx.list_id); changed.append(fmt_task(res.get("item") or item, schema, len(changed) + 1))
        elif intent == "reopen":
            if not config.has_permission(ctx, "complete"): raise PermissionError("You do not have permission to reopen action items.")
            res = slack_tools.reopen_action_item(item_id, ctx, ctx.list_id); changed.append(fmt_task(res.get("item") or item, schema, len(changed) + 1))
        else:
            # We must process the 'changes' list for multiple field updates if provided by Gemini
            changes = parsed.get("changes") or []
            if not changes:
                field = parsed.get("field"); value = parsed.get("value")
                if field == "name": value = parsed.get("new_name") or value
                if field: changes.append({"field": field, "value": value})
            if not changes: raise ValueError("Please specify what should change and the new value.")
            
            res = {}
            for change in changes:
                field = change.get("field"); value = change.get("value")
                if field in ("due_date", "date", "due"):
                    if value and value < current_date().isoformat():
                        raise ValueError("**Invalid due date**\n📅 The due date cannot be in the past.\n\nPlease provide a future date.")
                if field == "assignee" and parsed.get("assignee_self"): value = ctx.user_id
                if field == "priority": value = config.normalize_priority(value or parsed.get("priority"))
                if field == "status" and parsed.get("status"): value = parsed["status"]
                slack_tools.update_action_item_field(item_id, field, value, ctx, ctx.list_id)
            
            # Fetch the updated item to reflect actual values
            refreshed = slack_tools.list_action_items(ctx, ctx.list_id)
            updated_item = next((x for x in refreshed if slack_tools.extract_item_id(x) == item_id), item)
            changed.append(fmt_task(updated_item, schema, len(changed) + 1))
    verb = {"complete":"completed", "reopen":"reopened", "delete":"deleted", "update":"updated"}[intent]
    _pending.pop(memory_key, None)
    return f"*Action item{'s' if len(targets) > 1 else ''} {verb}*\n\n" + "\n\n".join(changed)


def process(text, user_id, channel_id, thread_ts=None):
    ctx = context(user_id, channel_id, thread_ts); key = f"{channel_id}:{user_id}:{thread_ts or 'root'}"
    logger.info("RAW_TEXT=%r user=%s channel=%s", text, user_id, channel_id)
    parsed = parse_intent(text)
    logger.info("PARSED intent=%s task_name=%r assignee=%r assignee_self=%s changes=%s",
                parsed.get("intent"), parsed.get("task_name"), parsed.get("assignee"),
                parsed.get("assignee_self"), parsed.get("changes"))
    if parsed.get("intent") == "temporarily_unavailable": return "The AI model is temporarily unavailable. Please try again in a few moments."
    if parsed.get("intent") == "out_of_scope": return None
    if parsed.get("intent") == "clarify": return parsed.get("clarification") or "Please clarify your action-item request."
    try:
        if parsed["intent"] == "create": return handle_create(parsed, ctx)
        if parsed["intent"] == "list": return handle_list(parsed, ctx, key)
        if parsed["intent"] in {"update", "complete", "reopen", "delete"}: return handle_mutation(parsed, ctx, key)
        return None
    except PermissionError as exc: return f"Permission denied: {exc}"
    except ValueError as exc: return str(exc)
    except Exception as exc:
        logger.exception("Request failed")
        return "I couldn't complete that action-item request because Slack returned an error."


def request_key(body, event, text):
    msg_id = event.get("client_msg_id")
    if msg_id: return msg_id
    eid = body.get("event_id") or event.get("event_ts") or event.get("ts")
    return str(eid or f"{event.get('channel')}:{event.get('user')}:{event.get('ts')}:{text}")


@app.command("/add")
def add_command(ack, body):
    ack()
    user = body.get("user_id"); channel = body.get("channel_id"); text = body.get("text", "").strip()
    key = "slash:" + str(body.get("trigger_id") or f"{channel}:{user}:{text}")
    if not claim_event(key): return
    if not text:
        post(channel, "Usage: `/add <action item>`"); return
    response = process(text, user, channel)
    if response: post(channel, response)


@app.event("app_mention")
def app_mention(body, event):
    if event.get("bot_id"): return
    # Mentions are handled ONLY here. Do not also process them through message events.
    key = request_key(body, event, event.get("text", ""))
    if not claim_event("event:" + key): return
    text = event.get("text", "")
    response = process(text, event.get("user"), event.get("channel"), event.get("thread_ts"))
    if response: post(event.get("channel"), response, event.get("thread_ts"))


@app.event({"type": "message", "subtype": None})
def dm_message(body, event):
    # Normal message events are accepted only for DMs. Channel messages are intentionally
    # ignored so an app_mention cannot be processed twice.
    if event.get("bot_id") or not event.get("user") or not str(event.get("channel", "")).startswith("D"):
        return
    key = request_key(body, event, event.get("text", ""))
    if not claim_event("event:" + key): return
    response = process(event.get("text", ""), event.get("user"), event.get("channel"), event.get("thread_ts"))
    if response: post(event.get("channel"), response, event.get("thread_ts"))


if __name__ == "__main__":
    logger.info("Slack List Assistant starting; Socket Mode enabled")
    SocketModeHandler(app, APP_TOKEN).start()
