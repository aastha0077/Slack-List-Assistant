import json
import logging
import os
import re
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
# In-memory write-through cache — fast reads, SQLite-backed for persistence
_pending = {}

OUT_OF_SCOPE = "I can only help with action items and Slack Lists."

# How long (seconds) to keep thread context: 3 hours
_CONTEXT_TTL = 10800


def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS events (key TEXT PRIMARY KEY, created REAL NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, status TEXT NOT NULL, created REAL NOT NULL)")
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
    """Write-through: update cache AND SQLite atomically."""
    _pending[key] = entry
    try:
        with _db_lock:
            conn = _db()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO thread_context(ctx_key, data, created) VALUES (?, ?, ?)",
                    (key, json.dumps(entry, default=str), entry.get("created", time.time()))
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as exc:
        logger.warning("ctx_write failed for key %r: %s", key, exc)


def _ctx_read(key: str) -> dict:
    """Read from cache first; fall back to SQLite if the in-memory entry is missing/expired."""
    v = _pending.get(key)
    if v and time.time() - v.get("created", 0) <= _CONTEXT_TTL:
        return v
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
        logger.warning("ctx_read failed for key %r: %s", key, exc)
    return {}


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


def context(user_id, channel_id, thread_ts=None, msg_ts=None):
    return config.build_context(user_id=user_id, channel_id=channel_id, thread_ts=thread_ts, msg_ts=msg_ts)


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


def make_context_keys(channel_id, thread_ts=None, msg_ts=None, user_id=None, bot_ts=None):
    """
    Generate prioritized lookup / storage keys for Slack thread and channel contexts.
    """
    keys = []
    # 1. Primary thread key (root thread timestamp)
    if thread_ts:
        keys.append(f"{channel_id}:{thread_ts}")
        if user_id:
            keys.append(f"{channel_id}:{user_id}:{thread_ts}")
    # 2. Bot response timestamp (if user starts a thread by replying directly to the bot's message)
    if bot_ts and bot_ts != thread_ts:
        keys.append(f"{channel_id}:{bot_ts}")
        if user_id:
            keys.append(f"{channel_id}:{user_id}:{bot_ts}")
    # 3. Triggering message timestamp
    if msg_ts and msg_ts != thread_ts and msg_ts != bot_ts:
        keys.append(f"{channel_id}:{msg_ts}")
        if user_id:
            keys.append(f"{channel_id}:{user_id}:{msg_ts}")
    # 4. Channel root fallback (ONLY when thread_ts is None)
    if not thread_ts:
        if user_id:
            keys.append(f"{channel_id}:{user_id}:root")
        keys.append(f"{channel_id}:root")
    return keys


def store_view(primary_key, items, schema=None, ctx=None, query_filter=None, secondary_keys=None):
    displayed_tasks = []
    if schema:
        for pos, item in enumerate(items, 1):
            item_id = slack_tools.extract_item_id(item)
            name = slack_tools.extract_item_name(item, schema)
            status = slack_tools.extract_status(item, schema)
            completed = slack_tools.extract_completed(item, schema)
            displayed_tasks.append({
                "position": pos,
                "item_id": item_id,
                "name": name,
                "status": status,
                "completed": completed,
                "raw_item": item,
            })

    entry = {
        "channel_id": ctx.channel_id if ctx else None,
        "thread_ts": ctx.thread_ts if ctx else None,
        "msg_ts": getattr(ctx, "msg_ts", None) if ctx else None,
        "bot_ts": None,
        "user_id": ctx.user_id if ctx else None,
        "displayed_tasks": displayed_tasks,
        "items": list(items),
        "timestamp": time.time(),
        "created": time.time(),
        "query_filter": query_filter,
    }

    all_keys = [primary_key] + (secondary_keys or [])
    for k in all_keys:
        if k:
            prev = _pending.get(k) or {}
            stored = dict(entry)
            stored["candidates"] = prev.get("candidates")
            _pending[k] = stored

    logger.info("STORE_VIEW saved %d item(s) under keys: %s", len(items), all_keys)

    # Keep memory bounded
    now = time.time()
    for k, v in list(_pending.items()):
        if now - v.get("created", 0) > 1800:
            _pending.pop(k, None)


def record_bot_response(channel_id, bot_ts, thread_ts=None, msg_ts=None, user_id=None):
    """
    Associate the newly sent bot response message timestamp with the active thread context.
    """
    if not bot_ts or not channel_id:
        return
    lookup_keys = make_context_keys(channel_id, thread_ts, msg_ts, user_id)
    entry = None
    for k in lookup_keys:
        if k in _pending:
            entry = _pending[k]
            break

    if entry:
        entry["bot_ts"] = bot_ts
        bot_keys = [f"{channel_id}:{bot_ts}"]
        if user_id:
            bot_keys.append(f"{channel_id}:{user_id}:{bot_ts}")
        for bk in bot_keys:
            _pending[bk] = entry
        logger.info("RECORD_BOT_RESPONSE attached bot_ts=%s to context keys: %s", bot_ts, bot_keys)


def get_thread_context(channel_id, thread_ts=None, msg_ts=None, user_id=None, primary_key=None):
    """
    Look up stored thread context with rich debug logging.
    """
    candidate_keys = []
    if primary_key:
        candidate_keys.append(primary_key)
    candidate_keys.extend(make_context_keys(channel_id, thread_ts, msg_ts, user_id))

    # If thread_ts was provided (user in a thread) but the thread was spawned from a root-channel
    # message, check the channel root context as fallback
    if thread_ts:
        if user_id:
            candidate_keys.append(f"{channel_id}:{user_id}:root")
        candidate_keys.append(f"{channel_id}:root")

    seen = set()
    ordered_keys = []
    for k in candidate_keys:
        if k and k not in seen:
            seen.add(k)
            ordered_keys.append(k)

    logger.info(
        "CONTEXT_LOOKUP: incoming channel_id=%s ts=%s thread_ts=%s user_id=%s keys_to_try=%s",
        channel_id, msg_ts, thread_ts, user_id, ordered_keys,
    )

    for k in ordered_keys:
        v = _pending.get(k)
        if v and time.time() - v.get("created", 0) <= 1800:
            displayed = v.get("displayed_tasks") or []
            items = v.get("items") or []
            logger.info(
                "CONTEXT_FOUND: matched_key=%s displayed_count=%d items_count=%d",
                k, len(displayed), len(items),
            )
            return v

    logger.warning("CONTEXT_NOT_FOUND: tried_keys=%s", ordered_keys)
    return None


def previous_items(key, ctx=None):
    ctx_data = get_thread_context(
        channel_id=ctx.channel_id if ctx else None,
        thread_ts=ctx.thread_ts if ctx else None,
        msg_ts=getattr(ctx, "msg_ts", None) if ctx else None,
        user_id=ctx.user_id if ctx else None,
        primary_key=key,
    )
    if ctx_data and "items" in ctx_data:
        return list(ctx_data["items"])
    v = _pending.get(key)
    return list(v["items"]) if v and time.time() - v.get("created", 0) <= 1800 and "items" in v else []


def selection_from(parsed, items, schema):
    if not items:
        return []

    # 1. Direct numeric selection index (-1 for last, 1-based positive index for 1st, 2nd, etc.)
    sel_idx = parsed.get("selection_index")
    if sel_idx is not None:
        try:
            sel_idx = int(sel_idx)
            if sel_idx == -1:
                return [items[-1]]
            elif 1 <= sel_idx <= len(items):
                return [items[sel_idx - 1]]
        except (ValueError, TypeError):
            pass

    # 2. Positional or set-based selections
    selection = (parsed.get("selection") or parsed.get("task_reference") or "").casefold().strip()

    if selection in {"last", "the last", "the last one", "final", "the final", "the final one", "previous", "the previous", "the previous one"}:
        return [items[-1]]
    if selection in {"first", "the first", "the first one", "#1", "1st"}:
        return items[:1]
    if selection in {"second", "the second", "the second one", "#2", "2nd"}:
        return [items[1]] if len(items) >= 2 else []
    if selection in {"third", "the third", "the third one", "#3", "3rd"}:
        return [items[2]] if len(items) >= 3 else []
    if selection in {"fourth", "the fourth", "the fourth one", "#4", "4th"}:
        return [items[3]] if len(items) >= 4 else []
    if selection in {"fifth", "the fifth", "the fifth one", "#5", "5th"}:
        return [items[4]] if len(items) >= 5 else []
    if selection == "single":
        if sel_idx == -1:
            return [items[-1]]
        return items[:1]
    if selection in {"all", "all of them", "everything"}:
        return items[:]
    if selection in {"both", "both of them"}:
        if len(items) > 2:
            names = "\n".join(f"• {slack_tools.extract_item_name(x, schema)}" for x in items[:8])
            raise ValueError(
                f"I found {len(items)} tasks. Which two tasks do you mean?\n" + names
            )
        return items[:2]

    # 3. Explicit selection numbers (e.g. #1, #3)
    nums = parsed.get("selection_numbers") or []
    if parsed.get("selection_count"):
        try:
            nums = list(range(1, int(parsed["selection_count"]) + 1))
        except (ValueError, TypeError):
            nums = []
    if nums:
        return [items[n - 1] for n in nums if isinstance(n, int) and 1 <= n <= len(items)]

    return []


def resolve_targets(parsed, items, schema, memory_key, ctx=None, intent=None):
    """
    Resolve the list item(s) that a mutation (update/delete/complete/reopen) should act on.

    `assignee` in parsed = the task's target user (used to NARROW the search scope).
    `ctx.user_id` = the REQUESTER (used for RBAC — checked by the caller, not here).

    These are different. An admin can delete any user's task; we still use the parsed
    assignee to find the right task when the command specifies one.
    """
    task_name = parsed.get("task_name")
    selection = (parsed.get("selection") or parsed.get("task_reference") or "").casefold()
    sel_idx = parsed.get("selection_index")

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

    is_positional = (
        task_name in {"__LAST__", "that one", "that task", "the one above", "last", "first", "second", "third"}
        or selection in {"single", "first", "second", "third", "last", "all", "both", "numbered"}
        or sel_idx is not None
    )

    if is_positional:
        if candidates:
            return selection_from(parsed, candidates, schema)

        ctx_data = get_thread_context(
            channel_id=ctx.channel_id if ctx else None,
            thread_ts=ctx.thread_ts if ctx else None,
            msg_ts=getattr(ctx, "msg_ts", None) if ctx else None,
            user_id=ctx.user_id if ctx else None,
            primary_key=memory_key,
        )

        if ctx_data:
            valid_prev = ctx_data.get("items") or [d["raw_item"] for d in ctx_data.get("displayed_tasks", []) if "raw_item" in d]
            current_ids = {slack_tools.extract_item_id(x) for x in items}
            valid_prev = [x for x in valid_prev if slack_tools.extract_item_id(x) in current_ids]
            if valid_prev:
                if target_assignee_id:
                    prev_filtered = [x for x in valid_prev if slack_tools.extract_assignee_id(x, schema) == target_assignee_id]
                    if prev_filtered:
                        selected_prev = selection_from(parsed, prev_filtered, schema)
                    else:
                        selected_prev = selection_from(parsed, valid_prev, schema)
                else:
                    selected_prev = selection_from(parsed, valid_prev, schema)

                # Return the live items matching the selected stored item IDs
                selected_ids = {slack_tools.extract_item_id(x) for x in selected_prev if slack_tools.extract_item_id(x)}
                targets = [x for x in items if slack_tools.extract_item_id(x) in selected_ids]
                return targets if targets else selected_prev

        # No valid context in this thread for positional reference — do not guess
        raise ValueError(
            "I couldn't find any recently displayed action items in this thread to reference. "
            "Please specify the name of the action item, or run 'show tasks' first."
        )

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

    # For completion, prefer uncompleted items if that resolves ambiguity
    if intent == "complete":
        pending_candidates = [m for m in all_matches if not slack_tools.extract_completed(m, schema)]
        if len(pending_candidates) == 1:
            return pending_candidates
        if len(pending_candidates) > 1:
            all_matches = pending_candidates

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
    _pending[memory_key] = {
        "items": previous_items(memory_key),
        "candidates": all_matches,
        "created": time.time(),
        "intent": intent,
        "parsed": parsed
    }
    names = "\n".join(f"• {slack_tools.extract_item_name(x, schema)}" for x in all_matches[:8])
    raise ValueError(
        f"Which {task_name} task do you mean?\n" + names
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
    if parsed.get("status") == "open":
        completed = False
    elif parsed.get("status") == "completed":
        completed = True
    elif parsed.get("all_tasks") or parsed.get("all"):
        completed = None
    elif completed is None:
        # Default for normal / general queries is Pending only
        completed = False

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

    secondary_keys = make_context_keys(ctx.channel_id, ctx.thread_ts, getattr(ctx, "msg_ts", None), ctx.user_id)
    store_view(memory_key, filtered, schema=schema, ctx=ctx, query_filter=parsed, secondary_keys=secondary_keys)

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

    # Assignee is optional — tasks can be created unassigned
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
    msg += f"\U0001f4cc *Task:* {task_name_out}\n"
    msg += f"\U0001f464 *Assignee:* {assignee_out}\n"
    if priority: msg += f"\U0001f534 *Priority:* {priority}\n"
    if due_date: msg += f"\U0001f4c5 *Due:* {due_date}\n"
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
                successes.append(result)
            else:
                failures.append(f"{i}. *{task_name}* — no response returned")
        except Exception as exc:
            logger.exception("handle_create_multi: error creating task %r", task_name)
            failures.append(f"{i}. *{task_name}* — {exc}")

    parts = []
    if successes:
        parts.append("\n\n".join(successes))
    if failures:
        failure_block = "*The following tasks could not be created:*\n" + "\n".join(failures)
        parts.append(failure_block)

    return "\n\n".join(parts) if parts else "No tasks were processed."



def handle_mutation(parsed, ctx, memory_key):
    # Multi-task mutation path (e.g. multiple tasks to complete or delete)
    tasks_list = parsed.get("tasks")
    if isinstance(tasks_list, list) and len(tasks_list) > 0:
        results = []
        for single_task in tasks_list:
            sub_parsed = dict(parsed)
            sub_parsed.pop("tasks", None)
            if isinstance(single_task, dict):
                sub_parsed.update(single_task)
            elif isinstance(single_task, str):
                sub_parsed["task_name"] = single_task
            try:
                res = handle_mutation(sub_parsed, ctx, memory_key)
                if res:
                    results.append(res)
            except Exception as e:
                task_label = sub_parsed.get("task_name") or "task"
                results.append(f"• *{task_label}*: {e}")
        return "\n\n".join(results) if results else "No action items were updated."

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
            if not config.has_permission(ctx, "delete"):
                raise PermissionError("You do not have permission to delete action items.")
            slack_tools.delete_action_item(item_id, ctx, ctx.list_id)
            changed.append(f"• {name}")
        elif intent == "complete":
            if not config.has_permission(ctx, "complete"):
                raise PermissionError("You do not have permission to complete action items.")
            if slack_tools.extract_completed(item, schema):
                changed.append(f"• *{name}* (already completed)")
            else:
                res = slack_tools.complete_action_item(item_id, ctx, ctx.list_id)
                refreshed = slack_tools.list_action_items(ctx, ctx.list_id)
                updated_item = next((x for x in refreshed if slack_tools.extract_item_id(x) == item_id), item)
                if not slack_tools.extract_completed(updated_item, schema):
                    changed.append(f"• *{name}* — ⚠️ Update requested, but task status in Slack List still shows pending.")
                else:
                    changed.append(fmt_task(updated_item, schema, len(changed) + 1))
        elif intent == "reopen":
            if not config.has_permission(ctx, "complete"):
                raise PermissionError("You do not have permission to reopen action items.")
            res = slack_tools.reopen_action_item(item_id, ctx, ctx.list_id)
            refreshed = slack_tools.list_action_items(ctx, ctx.list_id)
            updated_item = next((x for x in refreshed if slack_tools.extract_item_id(x) == item_id), item)
            changed.append(fmt_task(updated_item, schema, len(changed) + 1))
        else:
            # We must process the 'changes' list for multiple field updates
            changes = parsed.get("changes") or []
            if not changes:
                field = parsed.get("field")
                value = parsed.get("value")
                if field == "name":
                    value = parsed.get("new_name") or value
                if field:
                    changes.append({"field": field, "value": value})
            if not changes:
                raise ValueError("Please specify what should change and the new value.")

            for change in changes:
                field = change.get("field")
                value = change.get("value")
                if field in ("due_date", "date", "due"):
                    if value and value < current_date().isoformat():
                        raise ValueError("**Invalid due date**\n📅 The due date cannot be in the past.\n\nPlease provide a future date.")
                if field == "assignee" and parsed.get("assignee_self"):
                    value = ctx.user_id
                if field == "priority":
                    value = config.normalize_priority(value or parsed.get("priority"))
                if field == "status" and parsed.get("status"):
                    value = parsed["status"]
                slack_tools.update_action_item_field(item_id, field, value, ctx, ctx.list_id)

            # Fetch the updated item to reflect actual values
            refreshed = slack_tools.list_action_items(ctx, ctx.list_id)
            updated_item = next((x for x in refreshed if slack_tools.extract_item_id(x) == item_id), item)
            changed.append(fmt_task(updated_item, schema, len(changed) + 1))

    verb = {"complete": "completed", "reopen": "reopened", "delete": "deleted", "update": "updated"}[intent]
    if memory_key in _pending:
        _pending[memory_key].pop("candidates", None)
    return f"*Action item{'s' if len(targets) > 1 else ''} {verb}*\n\n" + "\n\n".join(changed)


def process(text, user_id, channel_id, thread_ts=None, msg_ts=None):
    ctx = context(user_id, channel_id, thread_ts, msg_ts)
    key = f"{channel_id}:{thread_ts}" if thread_ts else f"{channel_id}:{msg_ts or 'root'}"
    logger.info("RAW_TEXT=%r user=%s channel=%s thread_ts=%s msg_ts=%s", text, user_id, channel_id, thread_ts, msg_ts)

    # Check for pending thread clarification using get_thread_context
    pending = get_thread_context(channel_id, thread_ts, msg_ts, user_id, key)
    if pending and pending.get("candidates") and time.time() - pending.get("created", 0) <= 1800:
        candidates = pending["candidates"]
        schema = slack_tools.get_list_schema(ctx.list_id)
        t_clean = text.strip().casefold()
        matched_candidate = None

        if t_clean in ("1", "first", "#1"):
            matched_candidate = candidates[0] if candidates else None
        elif t_clean in ("2", "second", "#2") and len(candidates) > 1:
            matched_candidate = candidates[1]
        elif t_clean in ("3", "third", "#3") and len(candidates) > 2:
            matched_candidate = candidates[2]
        elif t_clean in ("last", "the last", "the last one") and candidates:
            matched_candidate = candidates[-1]
        elif t_clean in ("both", "all"):
            orig_parsed = dict(pending.get("parsed") or {})
            orig_intent = pending.get("intent") or orig_parsed.get("intent") or "complete"
            orig_parsed["intent"] = orig_intent
            orig_parsed["selection"] = t_clean
            pending.pop("candidates", None)
            return handle_mutation(orig_parsed, ctx, key)
        else:
            cand_matches = slack_tools.find_matches(candidates, text, schema)
            if len(cand_matches) == 1:
                matched_candidate = cand_matches[0]

        if matched_candidate:
            orig_parsed = dict(pending.get("parsed") or {})
            orig_intent = pending.get("intent") or orig_parsed.get("intent") or "complete"
            orig_parsed["intent"] = orig_intent
            orig_parsed["task_name"] = slack_tools.extract_item_name(matched_candidate, schema)
            pending.pop("candidates", None)
            return handle_mutation(orig_parsed, ctx, key)

    parsed = parse_intent(text)
    logger.info("PARSED intent=%s task_name=%r assignee=%r assignee_self=%s changes=%s",
                parsed.get("intent"), parsed.get("task_name"), parsed.get("assignee"),
                parsed.get("assignee_self"), parsed.get("changes"))
    if parsed.get("intent") == "temporarily_unavailable":
        return "The AI model is temporarily unavailable. Please try again in a few moments."
    if parsed.get("intent") == "out_of_scope":
        return None
    if parsed.get("intent") == "clarify":
        return parsed.get("clarification") or "Please clarify your action-item request."
    try:
        if parsed["intent"] == "create":
            # Multi-task path: parser returned a list of tasks
            tasks_list = parsed.get("tasks")
            if isinstance(tasks_list, list) and len(tasks_list) > 0:
                return handle_create_multi(tasks_list, ctx)
            # Single-task path: unchanged
            return handle_create(parsed, ctx)
        if parsed["intent"] == "list":
            return handle_list(parsed, ctx, key)
        if parsed["intent"] in {"update", "complete", "reopen", "delete"}:
            return handle_mutation(parsed, ctx, key)
        return None
    except PermissionError as exc:
        return f"Permission denied: {exc}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        logger.exception("Request failed")
        return "I couldn't complete that action-item request because Slack returned an error."


_MENTION_RE = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]+)?>", re.I)


def send_and_record(channel, text, thread_ts=None, user_id=None, msg_ts=None):
    resp = post(channel, text, thread_ts)
    bot_ts = None
    if resp:
        if isinstance(resp, dict):
            bot_ts = resp.get("ts")
        elif hasattr(resp, "get"):
            bot_ts = resp.get("ts")
        elif hasattr(resp, "data") and isinstance(resp.data, dict):
            bot_ts = resp.data.get("ts")
    if bot_ts:
        record_bot_response(channel_id=channel, bot_ts=bot_ts, thread_ts=thread_ts, msg_ts=msg_ts, user_id=user_id)
    return resp


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
    if response:
        send_and_record(channel, response, user_id=user)


@app.event("app_mention")
def app_mention(body, event):
    if event.get("bot_id"): return
    # Mentions are handled ONLY here. Do not also process them through message events.
    key = request_key(body, event, event.get("text", ""))
    if not claim_event("event:" + key): return
    text = event.get("text", "")
    response = process(
        text=text,
        user_id=event.get("user"),
        channel_id=event.get("channel"),
        thread_ts=event.get("thread_ts"),
        msg_ts=event.get("ts"),
    )
    if response:
        send_and_record(
            channel=event.get("channel"),
            text=response,
            thread_ts=event.get("thread_ts"),
            user_id=event.get("user"),
            msg_ts=event.get("ts"),
        )


@app.event({"type": "message", "subtype": None})
def message_handler(body, event):
    if event.get("bot_id") or not event.get("user"):
        return
    channel = str(event.get("channel", ""))
    thread_ts = event.get("thread_ts")
    is_dm = channel.startswith("D")

    text = event.get("text", "")
    # In public/private channels, if the bot is @mentioned, Slack fires app_mention. Skip here to avoid double-processing.
    if not is_dm and _MENTION_RE.search(text):
        return

    # In public/private channels without @mention, only process if inside a thread
    if not is_dm and not thread_ts:
        return

    key = request_key(body, event, text)
    if not claim_event("event:" + key): return

    response = process(
        text=text,
        user_id=event.get("user"),
        channel_id=event.get("channel"),
        thread_ts=thread_ts,
        msg_ts=event.get("ts"),
    )
    if response:
        send_and_record(
            channel=event.get("channel"),
            text=response,
            thread_ts=thread_ts,
            user_id=event.get("user"),
            msg_ts=event.get("ts"),
        )


if __name__ == "__main__":
    logger.info("Slack List Assistant starting; Socket Mode enabled")
    SocketModeHandler(app, APP_TOKEN).start()
