"""Validated mutation plans and consolidated business intelligence tools."""
from __future__ import annotations
from dataclasses import dataclass, field
from contextvars import ContextVar
from datetime import date, datetime
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from uuid import UUID
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo

from slack_sdk.errors import SlackApiError

from src import config
from src import slack_client as slack_tools
from src.slack_client import (
    complete_action_item, create_action_item, delete_action_item,
    get_list_schema, list_action_items, reopen_action_item,
    update_action_item_field,
)


logger = logging.getLogger(__name__)


@dataclass
class MutationResult:
    item_id: str
    item: dict | None = None
    verified: bool = False
    problems: list = field(default_factory=list)
    outcome: str = "unable_to_verify"


def authorize_collection(item_ids, current_items, intent, changes, ctx, schema):
    """Authorize and validate the complete exact-ID target set before writes."""
    if not config.has_permission(ctx, intent):
        raise PermissionError(f"You do not have permission to {intent} action items.")
    if len(item_ids) > 1 and not config.has_permission(ctx, "bulk"):
        raise PermissionError("Your role cannot perform bulk action-item operations.")
    live_ids = {slack_tools.extract_item_id(item) for item in current_items}
    if any(item_id not in live_ids for item_id in item_ids):
        raise ValueError("One or more selected action items no longer exist in the Slack List; no changes were made.")
    by_id = {slack_tools.extract_item_id(item): item for item in current_items}
    if not config.has_permission(ctx, "update_others") and any(
            ctx.user_id not in slack_tools.extract_assignee_ids(by_id[item_id], schema) for item_id in item_ids):
        raise PermissionError("Your role can only modify tasks assigned to you.")
    for change in changes:
        if change["field"] == "completed":
            allowed = config.has_permission(ctx, intent if intent in {"complete", "reopen"} else "complete")
        else:
            allowed = config.can_edit_field(ctx, change["field"])
        if not allowed:
            raise PermissionError(f"Your role cannot edit the {change['field']} field.")
        if change["field"] == "assignee":
            required = {"reassign" if slack_tools.extract_assignee_ids(by_id[item_id], schema) else "assign"
                        for item_id in item_ids}
            missing = [permission for permission in required if not config.has_permission(ctx, permission)]
            if missing:
                raise PermissionError(f"Your role cannot {missing[0]} action items.")
    logger.info(
        "permission_checked request_id=%s intent=%s operation=%s success=true item_count=%d",
        current_request_id(), intent, intent, len(item_ids))
    return tuple(item_ids)


def prepare_changes(parsed, ctx, schema, today):
    intent = parsed["intent"]
    if not config.has_permission(ctx, intent):
        raise PermissionError(f"You do not have permission to {intent} action items.")
    if intent == "delete":
        return []
    if intent in {"complete", "reopen"}:
        changes = [{"field": "completed", "value": intent == "complete"}]
    else:
        changes = parsed.get("changes") or []
        if not changes and parsed.get("field"):
            changes = [{"field": parsed["field"], "value": parsed.get("new_name") or parsed.get("value")}]
        if not changes:
            raise ValueError("Please specify what should change and the new value.")
    normalized = []
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("Each change must specify a field and value.")
        name, value = change.get("field"), change.get("value")
        name = {"due": "due_date", "date": "due_date", "deadline": "due_date",
                "title": "name", "task": "name", "owner": "assignee"}.get(name, name)
        allowed = (config.has_permission(ctx, intent if intent in {"complete", "reopen"} else "complete")
                   if name == "completed" else config.can_edit_field(ctx, name))
        if not allowed:
            raise PermissionError(f"Your role cannot edit the {name} field.")
        if name == "due_date":
            value = date.fromisoformat(str(value)).isoformat()
            if value < today.isoformat():
                raise ValueError("The due date cannot be in the past.")
        elif name == "priority":
            value = config.normalize_priority(value)
            if value not in {"P1", "P2", "P3"}:
                raise ValueError("Priority must be P1, P2 or P3.")
        elif name == "assignee":
            values = value if isinstance(value, list) else [value]
            resolved = []
            for candidate in values:
                user_id = ctx.user_id if str(candidate).casefold() in {"me", "myself"} else slack_tools.find_user_id(candidate)
                if not user_id:
                    raise ValueError(f"I couldn't resolve the new assignee {candidate!r}.")
                resolved.append(user_id)
            value = list(dict.fromkeys(resolved))
            if len(value) == 1:
                value = value[0]
            if not value:
                raise ValueError("I couldn't resolve the new assignee.")
        elif name == "name":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Task names cannot be empty.")
            value = value.strip()
        elif name == "status":
            value = config.normalize_status(value)
            if not value:
                raise ValueError("Please specify a valid task status.")
            # Completion and open share the canonical checkbox state.
            if value in {"open", "completed"}:
                name, value = "completed", value == "completed"
        if name == "completed" and not isinstance(value, bool):
            raise ValueError("Completed must be a boolean.")
        # Validate every cell before the first write, avoiding preventable partial updates.
        slack_tools._write_cell(schema, name, value)
        normalized.append({"field": name, "value": value})
    return normalized


def verify(item_id, changes, ctx, schema, deleted=False):
    result = MutationResult(item_id)
    logger.info(
        "verification_started request_id=%s operation=%s item_id=%s retry_count=0",
        current_request_id(), "delete" if deleted else "mutation", item_id)
    try:
        items = run(
            lambda: slack_tools.list_action_items(ctx, ctx.list_id), idempotent=True,
            operation_name="slack_verification_read")
    except Exception as exc:
        result.problems.append("verification read failed; outcome is unknown")
        logger.warning(
            "verification_completed request_id=%s operation=%s item_id=%s success=false error_category=%s",
            current_request_id(), "delete" if deleted else "mutation", item_id,
            classify(exc).category)
        return result
    result.item = next((x for x in items if slack_tools.extract_item_id(x) == item_id), None)
    if deleted:
        if result.item:
            result.problems.append("task still exists in Slack List")
    elif result.item is None:
        result.problems.append("selected task was not found during verification")
    else:
        extractors = {
            "name": slack_tools.extract_item_name, "priority": slack_tools.extract_priority,
            "assignee": slack_tools.extract_assignee_ids, "due_date": slack_tools.extract_due_date,
            "completed": slack_tools.extract_completed, "status": slack_tools.extract_status,
        }
        for change in changes:
            name, expected = change["field"], change["value"]
            extractor = extractors.get(name)
            actual = extractor(result.item, schema) if extractor else slack_tools.extract_field_value(result.item, schema, name)
            if name == "assignee":
                expected = expected if isinstance(expected, list) else [expected]
            if actual != expected:
                problem = "task status in Slack List still shows pending" if name == "completed" and expected is True else f"{name}: expected {expected!r}, found {actual!r}"
                result.problems.append(problem)
    result.verified = not result.problems
    result.outcome = "verified_success" if result.verified else (
        "partial_success" if result.item and 0 < len(result.problems) < len(changes) else "failed")
    logger.info(
        "verification_completed request_id=%s operation=%s item_id=%s success=%s error_category=%s",
        current_request_id(), "delete" if deleted else "mutation", item_id,
        str(result.verified).lower(), "none" if result.verified else "verification_error")
    return result


def execute(item_id, intent, changes, ctx, schema):
    key = checkpoint_key("mutation", {"list": ctx.list_id, "id": item_id, "intent": intent, "changes": changes})
    prior = checkpoint_read(key)
    if prior:
        observed = verify(item_id, changes, ctx, schema, deleted=intent == "delete")
        if observed.verified or prior["status"] == "verified":
            # Do not undo a later external edit when replaying a completed operation.
            return observed
    checkpoint_write(key, "started", {"item_id": item_id})
    write_error = False
    logger.info(
        "mutation_started request_id=%s intent=%s operation=%s item_id=%s retry_count=0",
        current_request_id(), intent, intent, item_id)
    try:
        if intent == "delete":
            # Delete is not retried because a timeout may occur after it commits.
            slack_tools.delete_action_item(item_id, ctx, ctx.list_id)
        elif intent == "complete":
            run(
                lambda: slack_tools.complete_action_item(item_id, ctx, ctx.list_id),
                idempotent=True, operation_name="slack_complete")
        elif intent == "reopen":
            run(
                lambda: slack_tools.reopen_action_item(item_id, ctx, ctx.list_id),
                idempotent=True, operation_name="slack_reopen")
        else:
            for change in changes:
                run(
                    lambda change=change: slack_tools.update_action_item_field(
                        item_id, change["field"], change["value"], ctx, ctx.list_id),
                    idempotent=True, operation_name=f"slack_update_{change['field']}")
    except Exception as exc:
        # A timeout can happen after Slack committed a write. Read back even on errors.
        write_error = True
        logger.warning(
            "mutation_completed request_id=%s intent=%s operation=%s item_id=%s success=false error_category=%s",
            current_request_id(), intent, intent, item_id,
            classify(exc).category)
    result = verify(item_id, changes, ctx, schema, deleted=intent == "delete")
    if write_error and not result.verified:
        result.problems.insert(0, "write failed or was interrupted; some fields may have changed")
    checkpoint_write(key, "verified" if result.verified else "unverified", {"item_id": item_id})
    logger.info(
        "mutation_completed request_id=%s intent=%s operation=%s item_id=%s success=%s error_category=%s",
        current_request_id(), intent, intent, item_id,
        str(result.verified).lower(), "none" if result.verified else "verification_error")
    return result


def execute_collection(item_ids, intent, changes, ctx, schema):
    """Execute and independently verify a pre-resolved exact-ID collection."""
    return [execute(item_id, intent, changes, ctx, schema) for item_id in item_ids]


_SECRET_NAME = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|app[_-]?token|bot[_-]?token|"
    r"oauth[_-]?token|client[_-]?secret|password)\b([\s:=\"']+)([^\s,;\"']+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_SLACK_TOKEN = re.compile(r"\bxox[a-z]-[A-Za-z0-9-]+\b", re.I)


def redact(value) -> str:
    """Remove credential-shaped values without logging configuration contents."""
    value_text = str(value or "")
    value_text = _BEARER.sub("Bearer [REDACTED]", value_text)
    value_text = _SLACK_TOKEN.sub("[REDACTED_SLACK_TOKEN]", value_text)
    value_text = _SECRET_NAME.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value_text
    )
    for name, secret in os.environ.items():
        if not secret or len(secret) < 8:
            continue
        if any(marker in name.upper() for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
            value_text = value_text.replace(secret, f"[REDACTED_{name.upper()}]")
    return value_text


def log_exception(logger: logging.Logger, event: str, exc: BaseException, **metadata) -> None:
    """Log a sanitized traceback for external service failures."""
    details = " ".join(f"{key}={redact(value)}" for key, value in metadata.items())
    safe_message = redact(str(exc)) or "(no exception message)"
    safe_exception = RuntimeError(safe_message).with_traceback(exc.__traceback__)
    logger.error(
        "%s%s exception_type=%s exception_message=%s",
        event,
        f" {details}" if details else "",
        type(exc).__name__,
        safe_message,
        exc_info=(RuntimeError, safe_exception, exc.__traceback__),
    )


_recovery_logger = logging.getLogger("error_recovery")
_request_id = ContextVar("request_id", default=None)


@dataclass(frozen=True)
class Failure:
    category: str
    recoverable: bool
    safe_to_retry: bool
    retry_count: int
    user_safe_message: str


class VerificationError(RuntimeError):
    """A mutation was attempted but its resulting Slack state was not confirmed."""


def begin_request(request_id=None):
    value = str(request_id or uuid.uuid4().hex)
    return value, _request_id.set(value)


def end_request(token):
    _request_id.reset(token)


def current_request_id():
    return _request_id.get() or "unscoped"


def _slack_status(exc):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    data = getattr(response, "data", None)
    if isinstance(response, dict):
        data = response
    error = str((data or {}).get("error") or "") if isinstance(data, dict) else ""
    retry_after = None
    headers = getattr(response, "headers", None) or {}
    try:
        retry_value = next((value for key, value in headers.items()
                            if str(key).casefold() == "retry-after"), None)
        retry_after = float(retry_value) if retry_value is not None else None
    except (TypeError, ValueError):
        pass
    return status, error.casefold(), retry_after


def classify(exc, retry_count=0):
    message = str(exc or "").casefold()
    status, slack_error, _ = _slack_status(exc)
    if isinstance(exc, PermissionError):
        return Failure("permission_denied", False, False, retry_count,
                       "You don't have permission to update this task.")
    if isinstance(exc, VerificationError):
        return Failure("verification_error", False, False, retry_count,
                       "The update was sent, but I couldn't verify the final task state, so I won't report it as successful.")
    if isinstance(exc, sqlite3.Error):
        return Failure("persistence_error", False, False, retry_count,
                       "I couldn't safely save the operation state. No successful change is being reported.")
    if type(exc).__name__ == "TranscriptionError" or "transcription" in type(exc).__module__:
        return Failure("transcription_error", False, False, retry_count,
                       "I couldn't transcribe the recording reliably. No task changes were made.")
    if isinstance(exc, SlackApiError) and (status == 429 or slack_error in {"ratelimited", "rate_limited"}):
        return Failure("slack_rate_limit", True, True, retry_count,
                       "Slack is temporarily rate limiting requests. No changes were confirmed.")
    if status == 429 or "rate limit" in message or "ratelimited" in message:
        return Failure("slack_rate_limit", True, True, retry_count,
                       "Slack is temporarily rate limiting requests. No changes were confirmed.")
    http_transient = isinstance(exc, HTTPError) and exc.code in {429, 500, 502, 503, 504}
    if (status in {500, 502, 503, 504} or http_transient
            or isinstance(exc, (TimeoutError, ConnectionError))
            or (isinstance(exc, URLError) and not isinstance(exc, HTTPError))
            or any(value in message for value in ("service_unavailable", "temporarily unavailable", "timed out", "timeout"))):
        return Failure("temporary_service_error", True, True, retry_count,
                       "I couldn't complete that update right now. No changes were confirmed.")
    if "duplicate" in message or "already exists" in message:
        return Failure("duplicate", False, False, retry_count,
                       "That task already exists. No duplicate was created.")
    if any(value in message for value in ("multiple matching", "which one", "ambiguous", "please clarify")):
        return Failure("ambiguity", False, False, retry_count,
                       "I found multiple matching tasks. Please specify which one.")
    if isinstance(exc, (ValueError, TypeError)):
        return Failure("validation_error", False, False, retry_count,
                       "That request contains an invalid value. No changes were made.")
    if any(value in message for value in ("ollama", "llm", "language model")):
        return Failure("llm_unavailable", True, True, retry_count,
                       "I couldn't understand that request right now. No task changes were made.")
    return Failure("unknown_error", False, False, retry_count,
                   "I couldn't complete that request safely. No changes were confirmed.")


def run(operation, *, idempotent=False, max_retries=2, sleeper=time.sleep,
        base_delay=0.25, operation_name="external_operation"):
    """Run an operation with retries only when classification and idempotency allow."""
    retry_count = 0
    while True:
        try:
            result = operation()
            if retry_count:
                _recovery_logger.info(
                    "recovery_completed request_id=%s operation=%s retry_count=%d success=true error_category=none",
                    current_request_id(), operation_name, retry_count)
            return result
        except Exception as exc:
            failure = classify(exc, retry_count)
            can_retry = bool(
                idempotent and failure.recoverable and failure.safe_to_retry
                and retry_count < max(0, int(max_retries)))
            _recovery_logger.warning(
                "recovery_started request_id=%s operation=%s retry_count=%d success=false error_category=%s retrying=%s error_type=%s message=%s",
                current_request_id(), operation_name, retry_count,
                failure.category, str(can_retry).lower(), type(exc).__name__, redact(exc))
            if not can_retry:
                _recovery_logger.warning(
                    "recovery_completed request_id=%s operation=%s retry_count=%d success=false error_category=%s",
                    current_request_id(), operation_name, retry_count, failure.category)
                raise
            _, _, retry_after = _slack_status(exc)
            delay = retry_after if retry_after is not None else base_delay * (2 ** retry_count)
            retry_count += 1
            sleeper(max(0.0, delay))


_active = ContextVar("slack_event", default=None)


def _json_default(value):
    """Convert supported application values only at the persistence boundary."""
    if isinstance(value, (date, datetime, UUID)):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    dictionary = getattr(value, "dict", None)
    if callable(dictionary):
        return dictionary()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(value, **kwargs):
    """Serialize checkpoint data through one controlled JSON boundary."""
    return json.dumps(value, default=_json_default, **kwargs)


def _delivery_connect(db):
    conn = db()
    conn.execute("CREATE TABLE IF NOT EXISTS delivery (key TEXT PRIMARY KEY, status TEXT NOT NULL, updated REAL NOT NULL, data TEXT NOT NULL)")
    conn.commit()
    return conn


def read(db, key):
    with _delivery_connect(db) as conn:
        row = conn.execute("SELECT status, updated, data FROM delivery WHERE key=?", (key,)).fetchone()
    conn.close()
    return {"status": row[0], "updated": row[1], **json.loads(row[2])} if row else None


def write(db, key, status, data):
    with _delivery_connect(db) as conn:
        conn.execute("INSERT OR REPLACE INTO delivery VALUES (?, ?, ?, ?)",
                     (key, status, time.time(), _json_dumps(data)))
    conn.close()


def claim(db, key):
    conn = _delivery_connect(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status, updated, data FROM delivery WHERE key=?", (key,)).fetchone()
        # The Socket Mode app serializes deliveries in-process. A non-final row
        # may be an interrupted request and must be resumed rather than dropped.
        if row and row[0] == "done":
            return None
        data = json.loads(row[2]) if row else {}
        data["resuming"] = bool(row)
        conn.execute("INSERT OR REPLACE INTO delivery VALUES (?, ?, ?, ?)",
                     (key, "processing", time.time(), _json_dumps(data)))
        conn.commit()
        return data
    finally:
        conn.close()


@contextmanager
def event(db, key):
    token = _active.set((db, key))
    try:
        yield
    finally:
        _active.reset(token)


def checkpoint_key(kind, identity):
    active = _active.get()
    if active is None:
        return None
    digest = hashlib.sha256(_json_dumps([kind, identity], sort_keys=True).encode()).hexdigest()
    return active[1] + ":operation:" + digest


def checkpoint_read(key):
    return read(_active.get()[0], key) if key else None


def checkpoint_write(key, status, data):
    if key:
        write(_active.get()[0], key, status, data)


def is_active():
    return _active.get() is not None


def staged_context():
    saved = checkpoint_read(checkpoint_key("context", "staged"))
    return saved.get("entries", {}) if saved else {}


def stage_context(key, entry):
    entries = staged_context()
    entries[key] = entry
    checkpoint_write(checkpoint_key("context", "staged"), "staged", {"entries": entries})


def stable_plan(identity, build):
    """Persist exact target IDs before the first write, including clarification."""
    key = checkpoint_key("plan", identity)
    saved = checkpoint_read(key)
    if saved:
        return saved["plan"]
    plan = build()
    checkpoint_write(key, "planned", {"plan": plan})
    return plan


def execute_event(db, key, run, send, recover_post=None, on_delivered=None):
    saved = claim(db, key)
    if saved is None:
        return
    token = _active.set((db, key))
    try:
        if "response" not in saved:
            with event(db, key):
                saved["response"] = run()
            write(db, key, "ready", saved)
        if saved["response"]:
            # If the previous post timed out after delivery, locate our metadata
            # marker in this conversation before sending another response.
            if saved.get("post_started") and recover_post:
                found = recover_post()
                if found:
                    saved["bot_ts"] = found
                    if on_delivered:
                        on_delivered()
                    write(db, key, "done", saved)
                    return
            saved["post_started"] = True
            write(db, key, "posting", saved)
            response = send(saved["response"])
            saved["bot_ts"] = response.get("ts") if response else None
        if on_delivered:
            on_delivered()
        write(db, key, "done", saved)
    except Exception:
        # A ready response is retained, so posting retries never replay mutations.
        write(db, key, "retryable", saved)
        raise
    finally:
        _active.reset(token)


# Physically migrated business and intelligence implementations.
from types import SimpleNamespace
audit_log = SimpleNamespace()
source_trace = SimpleNamespace()
workflow_safety = SimpleNamespace()
slack_presentation = SimpleNamespace()
visualization = SimpleNamespace()
progress_engine = SimpleNamespace()
project_intelligence = SimpleNamespace()
predictive_intelligence = SimpleNamespace()
operations_intelligence = SimpleNamespace()
control_tower = SimpleNamespace()
team_calendar = SimpleNamespace()
action_item_sentinel = SimpleNamespace()
smart_task_autopilot = SimpleNamespace()
command_center = SimpleNamespace()
agent_orchestrator = SimpleNamespace()
task_simulation = SimpleNamespace()
decision_ledger = SimpleNamespace()
deadline_reminders = SimpleNamespace()
visual_analytics = SimpleNamespace()
transcription = SimpleNamespace()
content_ingestion = SimpleNamespace()
references = SimpleNamespace()


# audit_log.py
'Best-effort append-only history for verified Slack List mutations.'
import json as _audit_log__json
audit_log.json = _audit_log__json
import sqlite3 as _audit_log__sqlite3
audit_log.sqlite3 = _audit_log__sqlite3
import time as _audit_log__time
audit_log.time = _audit_log__time
import uuid as _audit_log__uuid
audit_log.uuid = _audit_log__uuid
from src import slack_client as _audit_log__slack_tools
audit_log.slack_tools = _audit_log__slack_tools
def _audit_log___connect(db_path):
    conn = _audit_log__sqlite3.connect(db_path, timeout=10)
    conn.execute('\n        CREATE TABLE IF NOT EXISTS mutation_audit (\n            audit_id TEXT PRIMARY KEY,\n            created REAL NOT NULL,\n            team_id TEXT,\n            channel_id TEXT,\n            thread_ts TEXT,\n            actor_id TEXT,\n            actor_role TEXT,\n            list_id TEXT NOT NULL,\n            item_id TEXT NOT NULL,\n            operation TEXT NOT NULL,\n            changes_json TEXT NOT NULL,\n            before_json TEXT,\n            after_json TEXT\n        )\n    ')
    conn.execute('\n        CREATE TABLE IF NOT EXISTS audit_tracking (\n            list_id TEXT PRIMARY KEY,\n            started REAL NOT NULL\n        )\n    ')
    conn.commit()
    return conn
audit_log._connect = _audit_log___connect
def _audit_log__field_snapshot(item, schema):
    if item is None:
        return None
    return {'id': _audit_log__slack_tools.extract_item_id(item), 'name': _audit_log__slack_tools.extract_item_name(item, schema), 'assignees': _audit_log__slack_tools.extract_assignee_ids(item, schema), 'due_date': _audit_log__slack_tools.extract_due_date(item, schema), 'priority': _audit_log__slack_tools.extract_priority(item, schema), 'completed': _audit_log__slack_tools.extract_completed(item, schema), 'fields': item.get('fields') or item.get('cells') or []}
audit_log.field_snapshot = _audit_log__field_snapshot
def _audit_log__record(db_path, ctx, item_id, operation, changes, schema, before=None, after=None):
    with audit_log._connect(db_path) as conn:
        existing = conn.execute('SELECT MIN(created) FROM mutation_audit WHERE list_id=?', (ctx.list_id,)).fetchone()[0]
        conn.execute('INSERT OR IGNORE INTO audit_tracking VALUES (?, ?)', (ctx.list_id, existing or _audit_log__time.time()))
        conn.execute('INSERT INTO mutation_audit VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (str(_audit_log__uuid.uuid4()), _audit_log__time.time(), ctx.team_id, ctx.channel_id, ctx.thread_ts, ctx.user_id, ctx.role, ctx.list_id, item_id, operation, _audit_log__json.dumps(changes, default=str), _audit_log__json.dumps(audit_log.field_snapshot(before, schema), default=str) if before is not None else None, _audit_log__json.dumps(audit_log.field_snapshot(after, schema), default=str) if after is not None else None))
audit_log.record = _audit_log__record
def _audit_log__history(db_path, list_id, item_ids=(), limit=100, since=None, until=None):
    query = 'SELECT created, actor_id, actor_role, item_id, operation, changes_json, before_json, after_json FROM mutation_audit WHERE list_id=?'
    params = [list_id]
    if item_ids:
        placeholders = ','.join(('?' for _ in item_ids))
        query += f' AND item_id IN ({placeholders})'
        params.extend(item_ids)
    if since is not None:
        query += ' AND created>=?'
        params.append(float(since))
    if until is not None:
        query += ' AND created<?'
        params.append(float(until))
    query += ' ORDER BY created DESC LIMIT ?'
    params.append(limit)
    with audit_log._connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [{'created': row[0], 'actor_id': row[1], 'actor_role': row[2], 'item_id': row[3], 'operation': row[4], 'changes': _audit_log__json.loads(row[5]), 'before': _audit_log__json.loads(row[6]) if row[6] else None, 'after': _audit_log__json.loads(row[7]) if row[7] else None} for row in rows]
audit_log.history = _audit_log__history
def _audit_log__tracking_started(db_path, list_id):
    """Return the truthful start of locally available history for one List."""
    with audit_log._connect(db_path) as conn:
        row = conn.execute('SELECT started FROM audit_tracking WHERE list_id=?', (list_id,)).fetchone()
        if row:
            return row[0]
        earliest = conn.execute('SELECT MIN(created) FROM mutation_audit WHERE list_id=?', (list_id,)).fetchone()[0]
        started = earliest or _audit_log__time.time()
        conn.execute('INSERT OR IGNORE INTO audit_tracking VALUES (?, ?)', (list_id, started))
        return started
audit_log.tracking_started = _audit_log__tracking_started


# source_trace.py
'Persistent source evidence for tasks created from shared content.'
import json as _source_trace__json
source_trace.json = _source_trace__json
import sqlite3 as _source_trace__sqlite3
source_trace.sqlite3 = _source_trace__sqlite3
import time as _source_trace__time
source_trace.time = _source_trace__time
def _source_trace___connect(db_path):
    conn = _source_trace__sqlite3.connect(db_path, timeout=10)
    conn.execute('\n        CREATE TABLE IF NOT EXISTS task_sources (\n            list_id TEXT NOT NULL, item_id TEXT NOT NULL, created REAL NOT NULL,\n            source_type TEXT NOT NULL, source_reference TEXT,\n            confidence REAL, evidence TEXT, context_json TEXT NOT NULL,\n            PRIMARY KEY (list_id, item_id)\n        )\n    ')
    conn.commit()
    return conn
source_trace._connect = _source_trace___connect
def _source_trace__record(db_path, ctx, item_id, source):
    source = dict(source or {})
    context = {'team_id': ctx.team_id, 'channel_id': ctx.channel_id, 'thread_ts': ctx.thread_ts, 'actor_id': ctx.user_id}
    with source_trace._connect(db_path) as conn:
        conn.execute('INSERT OR REPLACE INTO task_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (ctx.list_id, item_id, _source_trace__time.time(), source.get('type', 'text'), source.get('reference'), source.get('confidence'), str(source.get('evidence') or '')[:240], _source_trace__json.dumps(context)))
source_trace.record = _source_trace__record
def _source_trace__get(db_path, list_id, item_id):
    with source_trace._connect(db_path) as conn:
        row = conn.execute('SELECT created, source_type, source_reference, confidence, evidence, context_json FROM task_sources WHERE list_id=? AND item_id=?', (list_id, item_id)).fetchone()
    if not row:
        return None
    return {'created': row[0], 'source_type': row[1], 'source_reference': row[2], 'confidence': row[3], 'evidence': row[4], 'context': _source_trace__json.loads(row[5])}
source_trace.get = _source_trace__get


# workflow_safety.py
'State fingerprints, duplicate detection and confirmation policy.'
import hashlib as _workflow_safety__hashlib
workflow_safety.hashlib = _workflow_safety__hashlib
import json as _workflow_safety__json
workflow_safety.json = _workflow_safety__json
import re as _workflow_safety__re
workflow_safety.re = _workflow_safety__re
import time as _workflow_safety__time
workflow_safety.time = _workflow_safety__time
from difflib import SequenceMatcher as _workflow_safety__SequenceMatcher
workflow_safety.SequenceMatcher = _workflow_safety__SequenceMatcher
from src import slack_client as _workflow_safety__slack_tools
workflow_safety.slack_tools = _workflow_safety__slack_tools
_workflow_safety__CONFIRMATION_TTL_SECONDS = 600
workflow_safety.CONFIRMATION_TTL_SECONDS = _workflow_safety__CONFIRMATION_TTL_SECONDS
_workflow_safety__DEFAULT_BULK_CONFIRMATION_THRESHOLD = 5
workflow_safety.DEFAULT_BULK_CONFIRMATION_THRESHOLD = _workflow_safety__DEFAULT_BULK_CONFIRMATION_THRESHOLD
def _workflow_safety___normalized_title(value):
    return _workflow_safety__re.sub('[^a-z0-9]+', ' ', str(value or '').casefold()).strip()
workflow_safety._normalized_title = _workflow_safety___normalized_title
def _workflow_safety__title_similarity(left, right):
    left_norm, right_norm = (workflow_safety._normalized_title(left), workflow_safety._normalized_title(right))
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    left_tokens, right_tokens = (set(left_norm.split()), set(right_norm.split()))
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = _workflow_safety__SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(jaccard, sequence)
workflow_safety.title_similarity = _workflow_safety__title_similarity
def _workflow_safety__likely_duplicates(name, items, schema, assignee_ids=(), threshold=0.92):
    """Return only strong candidates; similar but distinct titles remain valid."""
    wanted_assignees = set(assignee_ids or ())
    matches = []
    for item in items:
        if _workflow_safety__slack_tools.extract_completed(item, schema):
            continue
        score = workflow_safety.title_similarity(name, _workflow_safety__slack_tools.extract_item_name(item, schema))
        if score < threshold:
            continue
        existing_assignees = set(_workflow_safety__slack_tools.extract_assignee_ids(item, schema))
        if score < 1.0 and wanted_assignees and (existing_assignees != wanted_assignees):
            continue
        matches.append((score, item))
    return [item for _, item in sorted(matches, key=lambda pair: -pair[0])]
workflow_safety.likely_duplicates = _workflow_safety__likely_duplicates
def _workflow_safety__item_fingerprint(item, schema):
    value = {'id': _workflow_safety__slack_tools.extract_item_id(item), 'name': _workflow_safety__slack_tools.extract_item_name(item, schema), 'assignees': _workflow_safety__slack_tools.extract_assignee_ids(item, schema), 'due_date': _workflow_safety__slack_tools.extract_due_date(item, schema), 'priority': _workflow_safety__slack_tools.extract_priority(item, schema), 'completed': _workflow_safety__slack_tools.extract_completed(item, schema), 'fields': item.get('fields') or item.get('cells') or []}
    return _workflow_safety__hashlib.sha256(_workflow_safety__json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
workflow_safety.item_fingerprint = _workflow_safety__item_fingerprint
def _workflow_safety__snapshot_fingerprint(items, schema, item_ids=None):
    wanted = set(item_ids or ())
    selected = [item for item in items if not wanted or _workflow_safety__slack_tools.extract_item_id(item) in wanted]
    values = [(_workflow_safety__slack_tools.extract_item_id(item), workflow_safety.item_fingerprint(item, schema)) for item in selected]
    return _workflow_safety__hashlib.sha256(_workflow_safety__json.dumps(sorted(values), separators=(',', ':')).encode()).hexdigest()
workflow_safety.snapshot_fingerprint = _workflow_safety__snapshot_fingerprint
def _workflow_safety__confirmation_required(intent, item_count, changes, threshold=workflow_safety.DEFAULT_BULK_CONFIRMATION_THRESHOLD):
    if item_count < threshold:
        return False
    if intent == 'delete':
        return True
    high_impact = {'assignee', 'priority', 'due_date', 'completed', 'status'}
    return intent in {'update', 'complete', 'reopen'} and any((change.get('field') in high_impact for change in changes))
workflow_safety.confirmation_required = _workflow_safety__confirmation_required
def _workflow_safety__confirmation_is_fresh(confirmation, now=None):
    now = _workflow_safety__time.time() if now is None else now
    return bool(confirmation and confirmation.get('expires_at', 0) >= now)
workflow_safety.confirmation_is_fresh = _workflow_safety__confirmation_is_fresh


# slack_presentation.py
'Compact Slack-native presentation primitives with no task or query logic.'
import json as _slack_presentation__json
slack_presentation.json = _slack_presentation__json
import re as _slack_presentation__re
slack_presentation.re = _slack_presentation__re
from collections import Counter as _slack_presentation__Counter
slack_presentation.Counter = _slack_presentation__Counter
from dataclasses import dataclass as _slack_presentation__dataclass
slack_presentation.dataclass = _slack_presentation__dataclass
from datetime import date as _slack_presentation__date, timedelta as _slack_presentation__timedelta
slack_presentation.date = _slack_presentation__date
slack_presentation.timedelta = _slack_presentation__timedelta
from enum import Enum as _slack_presentation__Enum
slack_presentation.Enum = _slack_presentation__Enum
from html import escape as _slack_presentation__escape
slack_presentation.escape = _slack_presentation__escape
@_slack_presentation__dataclass(frozen=True)
class _slack_presentation__TaskRow:
    name: str
    assignee: str | None = None
    due_date: str | None = None
    show_due: bool = True
    priority: str | None = None
    status: str | None = None
    completed_date: str | None = None
    reviewer_attachments: tuple[tuple[str, str], ...] = ()
slack_presentation.TaskRow = _slack_presentation__TaskRow
class _slack_presentation__ResponseComplexity(str, _slack_presentation__Enum):
    SIMPLE = 'simple'
    STRUCTURED = 'structured'
    ANALYTICAL = 'analytical'
    COMPLEX = 'complex'
slack_presentation.ResponseComplexity = _slack_presentation__ResponseComplexity
class _slack_presentation__PresentationStrategy(str, _slack_presentation__Enum):
    COMPACT_CONFIRMATION = 'compact_confirmation'
    TASK_CARDS = 'task_cards'
    TASK_TABLE = 'task_table'
    SECTIONED_REPORT = 'sectioned_report'
    EMPTY_STATE = 'empty_state'
    ERROR_STATE = 'error_state'
slack_presentation.PresentationStrategy = _slack_presentation__PresentationStrategy
class _slack_presentation__ResponseType(str, _slack_presentation__Enum):
    TASK_LIST = 'task_list'
    TASK_SEARCH = 'task_search'
    COMPLETED_TASKS = 'completed_tasks'
    FOCUS = 'focus'
    ANALYTICS = 'analytics'
    EXPLANATION = 'explanation'
    SIMULATION = 'simulation'
    CONFIRMATION = 'confirmation'
    ERROR = 'error'
    EMPTY = 'empty'
slack_presentation.ResponseType = _slack_presentation__ResponseType
_slack_presentation__RESPONSE_TYPE_STRATEGIES = {slack_presentation.ResponseType.TASK_LIST: slack_presentation.PresentationStrategy.TASK_TABLE, slack_presentation.ResponseType.TASK_SEARCH: slack_presentation.PresentationStrategy.TASK_TABLE, slack_presentation.ResponseType.COMPLETED_TASKS: slack_presentation.PresentationStrategy.TASK_TABLE, slack_presentation.ResponseType.FOCUS: slack_presentation.PresentationStrategy.TASK_TABLE, slack_presentation.ResponseType.ANALYTICS: slack_presentation.PresentationStrategy.TASK_TABLE, slack_presentation.ResponseType.EXPLANATION: slack_presentation.PresentationStrategy.SECTIONED_REPORT, slack_presentation.ResponseType.SIMULATION: slack_presentation.PresentationStrategy.SECTIONED_REPORT, slack_presentation.ResponseType.CONFIRMATION: slack_presentation.PresentationStrategy.COMPACT_CONFIRMATION, slack_presentation.ResponseType.ERROR: slack_presentation.PresentationStrategy.ERROR_STATE, slack_presentation.ResponseType.EMPTY: slack_presentation.PresentationStrategy.EMPTY_STATE}
slack_presentation.RESPONSE_TYPE_STRATEGIES = _slack_presentation__RESPONSE_TYPE_STRATEGIES
def _slack_presentation__task_collection_strategy(count, *, grouped=False):
    """Choose a deterministic mobile-safe layout from response size."""
    if count <= 0:
        return (slack_presentation.ResponseComplexity.SIMPLE, slack_presentation.PresentationStrategy.EMPTY_STATE)
    return (slack_presentation.ResponseComplexity.STRUCTURED, slack_presentation.PresentationStrategy.TASK_TABLE)
slack_presentation.task_collection_strategy = _slack_presentation__task_collection_strategy
def _slack_presentation__text(value):
    """Escape Slack control characters while retaining renderer-owned mrkdwn."""
    return _slack_presentation__escape(str('' if value is None else value), quote=False)
slack_presentation.text = _slack_presentation__text
def _slack_presentation__notice(title, message, *, next_step=None):
    """Render a compact, Slack-native informational or failure notice."""
    lines = [f'*{slack_presentation.text(title)}*', '', slack_presentation.text(message)]
    if next_step:
        lines.extend(('', f'*Next step:* {slack_presentation.text(next_step)}'))
    return '\n'.join(lines)
slack_presentation.notice = _slack_presentation__notice
def _slack_presentation__empty_state(title, message, *, context=None):
    """Render one precise empty state without implying that all data is empty."""
    lines = [f'*{slack_presentation.text(title)}*', '', slack_presentation.text(message)]
    if context:
        lines.extend(('', slack_presentation.text(context)))
    return '\n'.join(lines)
slack_presentation.empty_state = _slack_presentation__empty_state
def _slack_presentation__permission_denied(message):
    """Render a user-facing authorization denial without implementation detail."""
    return slack_presentation.notice('Permission denied', message, next_step='Ask a workspace administrator if you need access to this action.')
slack_presentation.permission_denied = _slack_presentation__permission_denied
def _slack_presentation__clarification(message):
    """Render an actionable request for missing or ambiguous information."""
    return slack_presentation.notice('More information needed', message)
slack_presentation.clarification = _slack_presentation__clarification
def _slack_presentation__failure(message, *, next_step=None):
    """Render a verified-safe failure message; technical details belong in logs."""
    return slack_presentation.notice('Action not completed', message, next_step=next_step)
slack_presentation.failure = _slack_presentation__failure
def _slack_presentation__join_sections(*sections):
    """Join independently rendered Slack sections with one blank line.

    Callers provide semantic sections rather than managing boundary newlines.
    Internal line breaks inside task cards are preserved verbatim.
    """
    values = [str(section or '').strip() for section in sections if str(section or '').strip()]
    return '\n\n'.join(values)
slack_presentation.join_sections = _slack_presentation__join_sections
def _slack_presentation__render_section(title, *blocks):
    """Render a Slack-native heading and paragraph/item blocks."""
    heading = f'*{slack_presentation.text(title)}*' if title else ''
    return slack_presentation.join_sections(heading, *blocks)
slack_presentation.render_section = _slack_presentation__render_section
def _slack_presentation__render_item_list(items):
    """Keep multi-line task items visually separate on desktop and mobile."""
    return slack_presentation.join_sections(*(str(item or '').strip() for item in items))
slack_presentation.render_item_list = _slack_presentation__render_item_list
def _slack_presentation__render_task_card(name, metadata=(), *, prefix=None, detail=None):
    """Render one mobile-first task card with dominant title and quiet detail."""
    title = f'*{slack_presentation.text(name or 'Unnamed task')}*'
    if prefix:
        title = f'{slack_presentation.text(prefix)} {title}'
    lines = [title]
    indent = '   ' if prefix else '  '
    values = [slack_presentation.text(value) for value in metadata if value not in {None, ''}]
    if values:
        lines.append(indent + ' · '.join(values))
    if detail:
        lines.append(f'{indent}↳ {slack_presentation.text(detail)}')
    return '\n'.join(lines)
slack_presentation.render_task_card = _slack_presentation__render_task_card
def _slack_presentation__render_slack_table(columns, rows, *, title=None, summary=None, max_width=78):
    """Render comparable records as one bounded, Slack-safe monospace table.

    ``columns`` is an ordered collection of headings and ``rows`` contains
    sequences with matching positions. Values are escaped, flattened to one
    line and truncated only after the final mobile-width budget is known.
    """
    headings = [slack_presentation._table_cell(value) or '—' for value in columns]
    values = [[slack_presentation._table_cell(value) or '—' for value in row] for row in rows]
    if not headings:
        raise ValueError('A Slack table needs at least one column.')
    if any((len(row) != len(headings) for row in values)):
        raise ValueError('Slack table rows must match the declared columns.')
    natural = [max(len(headings[index]), *(len(row[index]) for row in values)) for index in range(len(headings))]
    minimum = [max(len(heading), 12 if index == 0 else len(heading)) for index, heading in enumerate(headings)]
    widths = list(natural)
    spacing = 2 * max(0, len(widths) - 1)
    while sum(widths) + spacing > max_width:
        candidates = [index for index, width in enumerate(widths) if width > minimum[index]]
        if not candidates:
            break
        widest = max(candidates, key=lambda index: (widths[index] - minimum[index], widths[index]))
        widths[widest] -= 1

    def truncate(value, width):
        return value if len(value) <= width else value[:max(1, width - 1)].rstrip() + '…'

    def render(row):
        return '  '.join((truncate(value, widths[index]).ljust(widths[index]) for index, value in enumerate(row))).rstrip()
    table = '\n'.join(('```', render(headings), '─' * min(max_width, sum(widths) + spacing), *(render(row) for row in values), '```'))
    sections = []
    if title:
        sections.append(f'*{slack_presentation.text(title)}*')
    if summary:
        sections.append(f'*{slack_presentation.text(summary)}*')
    sections.append(table)
    return slack_presentation.join_sections(*sections)
slack_presentation.render_slack_table = _slack_presentation__render_slack_table
def _slack_presentation__focus_reason(value):
    """Turn deterministic risk language into concise display language."""
    reason = str(value or '').strip()
    match = _slack_presentation__re.fullmatch('P1 \\+ overdue by (\\d+) days?', reason, _slack_presentation__re.I)
    if match:
        days = int(match.group(1))
        return f'P1 priority + {days} day{('s' if days != 1 else '')} overdue'
    match = _slack_presentation__re.fullmatch('overdue by (\\d+) days?', reason, _slack_presentation__re.I)
    if match:
        days = int(match.group(1))
        return f'{days} day{('s' if days != 1 else '')} overdue'
    return reason[:1].upper() + reason[1:] if reason else ''
slack_presentation.focus_reason = _slack_presentation__focus_reason
def _slack_presentation__validate_slack_response(value, *, resolve_user=None):
    """Return Slack-safe text and non-sensitive response-quality issue codes.

    This final presentation gate only repairs defects that are safe to correct
    without interpreting task intent or changing business data. Structural
    concerns that require feature-specific judgment are reported for logs and
    regression tests, but are not silently rewritten.
    """
    rendered = str(value or '').strip()
    issues = []
    if not rendered:
        return (slack_presentation.failure('No response content was available.', next_step='Please try the request again.'), ('empty_response',))
    # Slack's mrkdwn control syntax is not HTML. Decode only common escaped
    # text entities here; keep angle brackets non-syntactic so an escaped
    # task title cannot turn into a mention or clickable Slack control token.
    entity_replacements = {'&#x20;': ' ', '&#32;': ' ', '&amp;': '&',
                           '&lt;': '‹', '&gt;': '›'}
    for entity, replacement in entity_replacements.items():
        if entity in rendered:
            issues.append('html_entity')
            rendered = rendered.replace(entity, replacement)
    rendered = '\n'.join(line.rstrip() for line in rendered.splitlines())
    if 'Traceback (most recent call last):' in rendered or _slack_presentation__re.search('<[^>]+ object at 0x[0-9a-f]+>', rendered, _slack_presentation__re.I):
        return (slack_presentation.failure("I couldn't safely present the result of that request.", next_step='Please try again. Technical details were recorded in the application logs.'), ('internal_debug_output',))
    try:
        decoded = _slack_presentation__json.loads(rendered)
    except (TypeError, ValueError, _slack_presentation__json.JSONDecodeError):
        decoded = None
    if isinstance(decoded, (dict, list)):
        return (slack_presentation.failure("I couldn't safely present the structured result of that request.", next_step='Please try again.'), ('raw_json',))
    if _slack_presentation__re.search('\\\\\\*\\\\\\*[^\\n]+?\\\\\\*\\\\\\*', rendered):
        issues.append('escaped_markdown_bold')
        rendered = _slack_presentation__re.sub('\\\\\\*\\\\\\*([^\\n]+?)\\\\\\*\\\\\\*', '*\\1*', rendered)
    if _slack_presentation__re.search('\\*\\*[^*\\n]+\\*\\*', rendered):
        issues.append('markdown_bold')
        rendered = _slack_presentation__re.sub('\\*\\*([^*\\n]+)\\*\\*', '*\\1*', rendered)
    if _slack_presentation__re.search('\\[:red_circle:\\]\\(https?://[^)]+\\)', rendered, _slack_presentation__re.I):
        issues.append('emoji_image_link')
        rendered = _slack_presentation__re.sub('\\[:red_circle:\\]\\(https?://[^)]+\\)', '🔴', rendered, flags=_slack_presentation__re.I)
    if ':robot_face:' in rendered or '🤖' in rendered:
        issues.append('decorative_robot')
        rendered = rendered.replace(':robot_face:', '').replace('🤖', '').strip()

    def replace_user(match):
        user_id = match.group(0)
        name = resolve_user(user_id) if resolve_user else None
        issues.append('raw_user_id')
        return str(name or 'Workspace member')
    rendered = _slack_presentation__re.sub('(?<!<@)\\bU[A-Z0-9]{8,}\\b', replace_user, rendered)
    if _slack_presentation__re.search('\\bF[A-Z0-9]{8,}\\b', rendered):
        issues.append('raw_list_id')
        rendered = _slack_presentation__re.sub('\\bF[A-Z0-9]{8,}\\b', 'Action Items', rendered)
    record_id = '\\b(?:REC|I)(?=[A-Z0-9]*\\d)[A-Z0-9]{8,}\\b'
    if _slack_presentation__re.search(record_id, rendered, _slack_presentation__re.I):
        issues.append('raw_record_id')
        rendered = _slack_presentation__re.sub(record_id, 'Task record', rendered, flags=_slack_presentation__re.I)
    if rendered.count('```') % 2:
        issues.append('unclosed_code_fence')
        rendered += '\n```'
    headings = [line.strip() for line in rendered.splitlines() if _slack_presentation__re.fullmatch('\\*[^*\\n]+\\*', line.strip())]
    if len(headings) != len(set(headings)):
        issues.append('duplicate_section')
    lines = rendered.splitlines()
    for index, line in enumerate(lines):
        if _slack_presentation__re.fullmatch('\\*[^*\\n]+\\*', line.strip()):
            following = next((candidate.strip() for candidate in lines[index + 1:] if candidate.strip()), '')
            if following and _slack_presentation__re.fullmatch('\\*[^*\\n]+\\*', following):
                issues.append('empty_section')
                break
    if any((line.count('*') % 2 for line in lines if not line.strip().startswith('```') and '*' in line)):
        issues.append('broken_slack_formatting')
    lower = rendered.casefold()
    if ('no action items found' in lower or 'no tasks found' in lower) and _slack_presentation__re.search('\\b[1-9]\\d*\\s+(?:pending|completed|authorized)\\b', lower):
        issues.append('contradictory_empty_state')
    if len(rendered) > 12000:
        issues.append('excessive_length')
    return (rendered, tuple(dict.fromkeys(issues)))
slack_presentation.validate_slack_response = _slack_presentation__validate_slack_response
def _slack_presentation___date_value(value):
    if not value:
        return None
    try:
        return _slack_presentation__date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None
slack_presentation._date_value = _slack_presentation___date_value
def _slack_presentation__compact_date(value, today=None):
    if not value:
        return 'No due date'
    parsed = slack_presentation._date_value(value)
    if parsed is None:
        return slack_presentation.text(value)
    today = today or _slack_presentation__date.today()
    label = parsed.strftime('%b %d').replace(' 0', ' ')
    return f'{label}, {parsed.year}'
slack_presentation.compact_date = _slack_presentation__compact_date
def _slack_presentation__compact_priority(value):
    """Keep configured priority values intact and shorten an unset priority."""
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.casefold() in {'', 'none', 'not set', 'no priority', 'no priority assigned', '—'}:
        return '—'
    return normalized
slack_presentation.compact_priority = _slack_presentation__compact_priority
def _slack_presentation__due_label(row: slack_presentation.TaskRow, today=None):
    """Present a due date as a compact date or an immediately useful time state."""
    if not row.show_due:
        return None
    if not row.due_date:
        return 'No due date'
    parsed = slack_presentation._date_value(row.due_date)
    if parsed is None:
        return slack_presentation.text(row.due_date)
    today = today or _slack_presentation__date.today()
    if str(row.status or '').casefold() != 'completed':
        if parsed < today:
            return f'🔴 Overdue ({slack_presentation.compact_date(row.due_date, today)})'
        if parsed == today:
            return '🟡 Due today'
        if parsed == today + _slack_presentation__timedelta(days=1):
            return 'Due tomorrow'
    return slack_presentation.compact_date(row.due_date, today)
slack_presentation.due_label = _slack_presentation__due_label
def _slack_presentation___status_is_redundant(status, due):
    return str(status or '').casefold() == 'pending' and due in {'🟡 Due today', 'Due tomorrow'} or (str(status or '').casefold() == 'pending' and str(due or '').startswith('🔴 Overdue'))
slack_presentation._status_is_redundant = _slack_presentation___status_is_redundant
def _slack_presentation__task_summary(row: slack_presentation.TaskRow, *, show_assignee=True, show_due=True, show_priority=True, show_status=True, today=None):
    """Render task content without a bullet, suitable for any response type."""
    due = slack_presentation.due_label(row, today) if show_due else None
    details = []
    if show_priority and row.priority is not None:
        details.append(slack_presentation.compact_priority(row.priority))
    if show_assignee and row.assignee:
        details.append(row.assignee)
    if due:
        details.append(due)
    if show_status and row.status and (not slack_presentation._status_is_redundant(row.status, due)):
        details.append(row.status)
    suffix = ' · '.join((slack_presentation.text(value) for value in details if value is not None))
    return f'*{slack_presentation.text(row.name or 'Unnamed task')}*' + (f' · {suffix}' if suffix else '')
slack_presentation.task_summary = _slack_presentation__task_summary
def _slack_presentation__task_line(row: slack_presentation.TaskRow, position=None, bullet=False, *, show_assignee=True, show_due=True, show_priority=True, show_status=True, today=None):
    prefix = '•' if bullet or position is None else f'{position}.'
    return f'{prefix} ' + slack_presentation.task_summary(row, show_assignee=show_assignee, show_due=show_due, show_priority=show_priority, show_status=show_status, today=today)
slack_presentation.task_line = _slack_presentation__task_line
def _slack_presentation___title_rules(title, rows, show_assignee, show_due, show_status):
    normalized = str(title or '').casefold()
    if show_assignee is None:
        show_assignee = not normalized.startswith(('your ', 'my ', 'focus today', "today's focus"))
    if show_due is None:
        show_due = not any((value in normalized for value in ('focus today', "today's focus", 'due today', 'due tomorrow')))
    if show_status is None:
        statuses = {str(row.status or '').casefold() for row in rows if row.status}
        status_is_heading = statuses == {'pending'} and any((value in normalized for value in ('pending', 'focus today', "today's focus", 'due today', 'due tomorrow', 'overdue'))) or (statuses == {'completed'} and 'completed' in normalized)
        owner_implied = normalized.startswith(('your ', 'my ')) and statuses == {'pending'}
        show_status = not (status_is_heading or owner_implied)
    return (show_assignee, show_due, show_status)
slack_presentation._title_rules = _slack_presentation___title_rules
def _slack_presentation___summary(rows, title, today):
    if len(rows) < 4:
        return ''
    values = []
    priorities = _slack_presentation__Counter((slack_presentation.compact_priority(row.priority) for row in rows))
    named_priorities = [priority for priority in ('P1', 'P2', 'P3') if priorities[priority]]
    if len(named_priorities) > 1:
        values.extend((f'{priorities[priority]} {priority}' for priority in named_priorities))
    normalized_title = str(title or '').casefold()
    if 'overdue' not in normalized_title:
        overdue = sum((bool(slack_presentation._date_value(row.due_date) and slack_presentation._date_value(row.due_date) < today) and str(row.status or '').casefold() != 'completed' for row in rows))
        if overdue:
            values.append(f'{overdue} overdue')
    if 'today' not in normalized_title:
        due_today = sum((slack_presentation._date_value(row.due_date) == today and str(row.status or '').casefold() != 'completed' for row in rows))
        if due_today:
            values.append(f'{due_today} due today')
    return ' · '.join(values)
slack_presentation._summary = _slack_presentation___summary
def _slack_presentation___due_group(row, today):
    if str(row.status or '').casefold() == 'completed':
        return 'Completed'
    parsed = slack_presentation._date_value(row.due_date)
    if parsed is None:
        return 'No Due Date'
    if parsed < today:
        return 'Overdue'
    if parsed == today:
        return 'Due Today'
    return 'Upcoming'
slack_presentation._due_group = _slack_presentation___due_group
def _slack_presentation___table_cell(value):
    """Keep untrusted values on one line without breaking Slack's code fence."""
    return slack_presentation.text(value).replace('\r', ' ').replace('\n', ' ').replace('```', "''' ").strip()
slack_presentation._table_cell = _slack_presentation___table_cell
def _slack_presentation___table_due_label(row, today):
    if not row.show_due:
        return None
    parsed = slack_presentation._date_value(row.due_date)
    if not parsed:
        return '—'
    label = slack_presentation.compact_date(row.due_date, today)
    if str(row.status or '').casefold() != 'completed':
        if parsed < today:
            return f'🔴 {label}'
        if parsed == today:
            return '🟡 Today'
    return label
slack_presentation._table_due_label = _slack_presentation___table_due_label
def _slack_presentation___task_table(rows, *, show_assignee, show_due, show_status, today):
    """Render a compact, reliably aligned Slack mrkdwn table."""

    def task_name(row):
        name = row.name or 'Unnamed task'
        return f'{name} ✓' if str(row.status or '').casefold() == 'completed' else name
    columns = [('Task', task_name)]
    if any((row.priority is not None for row in rows)):
        columns.append(('Priority', lambda row: slack_presentation.compact_priority(row.priority) or '—'))
    if show_assignee:
        columns.append(('Owner', lambda row: row.assignee or 'Unassigned'))
    completed_only = bool(rows) and all((str(row.status or '').casefold() == 'completed' for row in rows))
    if completed_only:
        columns = [('Task', lambda row: row.name or 'Unnamed task'), ('Completed', lambda row: slack_presentation.compact_date(row.completed_date, today) if row.completed_date else '✓')]
    else:
        if show_due:
            columns.append(('Due', lambda row: slack_presentation._table_due_label(row, today) or '—'))
        if show_status:
            columns.append(('Status', lambda row: row.status or '—'))
    values = [[getter(row) for _, getter in columns] for row in rows]
    return slack_presentation.render_slack_table([name for name, _ in columns], values)
slack_presentation._task_table = _slack_presentation___task_table
def _slack_presentation___collection_heading(title):
    normalized = str(title or '').casefold()
    if 'search' in normalized:
        return '🔎 Task Search'
    if 'current list state' in normalized:
        return '✅ Completed (current List state)'
    if 'completed' in normalized:
        return '✅ Completed Tasks'
    if normalized.startswith(('your ', 'my ')):
        return '📋 My Action Items'
    return f'📋 {title or 'Action Items'}'
slack_presentation._collection_heading = _slack_presentation___collection_heading
def _slack_presentation___collection_summary(rows, title, today):
    count = len(rows)
    normalized = str(title or '').casefold()
    statuses = {str(row.status or '').casefold() for row in rows if row.status}
    if 'search' in normalized:
        first = f'{count} match{('es' if count != 1 else '')}'
    elif statuses == {'completed'} or 'completed' in normalized:
        first = f'{count} completed'
    elif statuses == {'pending'} or 'pending' in normalized or normalized.startswith(('your ', 'my ')):
        first = f'{count} pending'
    else:
        first = f'{count} task{('s' if count != 1 else '')}'
    overdue = sum((bool(slack_presentation._date_value(row.due_date) and slack_presentation._date_value(row.due_date) < today) and str(row.status or '').casefold() != 'completed' for row in rows))
    due_today = sum((slack_presentation._date_value(row.due_date) == today and str(row.status or '').casefold() != 'completed' for row in rows))
    values = [first]
    if overdue:
        values.append(f'{overdue} overdue')
    if due_today:
        values.append(f'{due_today} due today')
    return ' · '.join(values)
slack_presentation._collection_summary = _slack_presentation___collection_summary
def _slack_presentation___collection_metrics(rows, title, today):
    if str(title or '').strip().casefold() != 'action items':
        return ''
    pending = sum((str(row.status or '').casefold() != 'completed' for row in rows))
    priorities = _slack_presentation__Counter((slack_presentation.compact_priority(row.priority) for row in rows))
    overdue = sum((bool(slack_presentation._date_value(row.due_date) and slack_presentation._date_value(row.due_date) < today) and str(row.status or '').casefold() != 'completed' for row in rows))
    unassigned = sum((str(row.assignee or '').strip().casefold() in {'', 'unassigned'} for row in rows))
    return '\n'.join((f'• Total: {len(rows)} · Pending: {pending} · Completed: {len(rows) - pending}', f'• P1: {priorities['P1']} · P2: {priorities['P2']} · P3: {priorities['P3']}', f'• Overdue: {overdue} · Unassigned: {unassigned}'))
slack_presentation._collection_metrics = _slack_presentation___collection_metrics
def _slack_presentation___task_cards(indexed_rows, *, show_assignee, show_due, show_status, today):
    """Render small collections as readable mobile-friendly task cards."""
    blocks = []
    for position, row in indexed_rows:
        completed = str(row.status or '').casefold() == 'completed'
        name = f'{row.name or 'Unnamed task'} ✓' if completed else row.name or 'Unnamed task'
        details = []
        if row.priority is not None:
            details.append(slack_presentation.compact_priority(row.priority) or '—')
        if show_assignee:
            details.append(row.assignee or 'Unassigned')
        due = slack_presentation.due_label(row, today) if show_due else None
        if due:
            details.append(due)
        if show_status and row.status and (not completed) and (not slack_presentation._status_is_redundant(row.status, due)):
            details.append(row.status)
        blocks.append(slack_presentation.render_task_card(name, details, prefix=f'{position}.'))
    return blocks
slack_presentation._task_cards = _slack_presentation___task_cards
def _slack_presentation__task_collection(rows, title='Action Items', numbered=None, *, show_assignee=None, show_due=None, show_status=None, summary=True, group_due=False, group_status=False, empty_message=None, today=None, strategy=None):
    """Render an adaptive, mobile-readable task collection."""
    rows = list(rows)
    count = len(rows)
    heading = slack_presentation._collection_heading(title)
    header = f'*{slack_presentation.text(heading)}*'
    if not rows:
        return header + '\n\n' + (empty_message or 'No authorized action items found.')
    today = today or _slack_presentation__date.today()
    show_assignee, show_due, show_status = slack_presentation._title_rules(title, rows, show_assignee, show_due, show_status)
    summary_text = slack_presentation._collection_summary(rows, title, today) if summary else ''
    metrics = slack_presentation._collection_metrics(rows, title, today) if summary and group_status else ''
    useful_due_grouping = group_due and count >= 3 and any((slack_presentation._due_group(row, today) in {'Overdue', 'Due Today'} for row in rows))
    grouping_requested = useful_due_grouping or group_status
    if grouping_requested:
        labels = ('Overdue', 'Due Today', 'Upcoming', 'No Due Date', 'Pending', 'Completed')
        rank = {label: index for index, label in enumerate(labels)}
        bucket_labels = [slack_presentation._due_group(row, today) if useful_due_grouping else 'Completed' if str(row.status or '').casefold() == 'completed' else 'Pending' for row in rows]
        grouping_requested = [rank[label] for label in bucket_labels] == sorted((rank[label] for label in bucket_labels))
    if not grouping_requested:
        selected_strategy = slack_presentation.PresentationStrategy(strategy) if strategy else slack_presentation.PresentationStrategy.TASK_TABLE
        if selected_strategy == slack_presentation.PresentationStrategy.TASK_CARDS:
            body = slack_presentation.render_item_list(slack_presentation._task_cards(list(enumerate(rows, 1)), show_assignee=show_assignee, show_due=show_due, show_status=show_status, today=today))
        else:
            body = slack_presentation._task_table(rows, show_assignee=show_assignee, show_due=show_due, show_status=show_status and 'search' in str(title).casefold(), today=today)
        return slack_presentation.join_sections(header, f'*{summary_text}*' if summary_text else None, slack_presentation.render_section('Summary', metrics) if metrics else None, body)
    sections = [header, f'*{summary_text}*' if summary_text else None, slack_presentation.render_section('Summary', metrics) if metrics else None]
    groups = {label: [] for label in labels}
    for index, row in enumerate(rows, 1):
        label = slack_presentation._due_group(row, today) if useful_due_grouping else 'Completed' if str(row.status or '').casefold() == 'completed' else 'Pending'
        groups[label].append((index, row))
    for label, group in groups.items():
        if not group:
            continue
        hide_due = label in {'Overdue', 'Due Today'}
        selected_strategy = slack_presentation.PresentationStrategy(strategy) if strategy else slack_presentation.PresentationStrategy.TASK_TABLE
        if selected_strategy == slack_presentation.PresentationStrategy.TASK_CARDS:
            body = slack_presentation.render_item_list(slack_presentation._task_cards(group, show_assignee=show_assignee, show_due=show_due and (not hide_due), show_status=False, today=today))
        else:
            body = slack_presentation._task_table([row for _, row in group], show_assignee=show_assignee, show_due=show_due and (not hide_due), show_status=False, today=today)
        sections.append(slack_presentation.render_section(label, body))
    return slack_presentation.join_sections(*sections)
slack_presentation.task_collection = _slack_presentation__task_collection
def _slack_presentation__task_detail(row: slack_presentation.TaskRow, bullet=True):
    """Compatibility helper for a labeled single-task detail view."""
    prefix = '• ' if bullet else ''
    lines = [f'{prefix}*{slack_presentation.text(row.name or 'Unnamed task')}*']
    fields = [('Assignee', row.assignee), ('Due', slack_presentation.compact_date(row.due_date) if row.show_due else None), ('Priority', slack_presentation.compact_priority(row.priority)), ('Status', row.status)]
    lines.extend((f'  • {label}: {slack_presentation.text(value)}' for label, value in fields if value is not None))
    return '\n'.join(lines)
slack_presentation.task_detail = _slack_presentation__task_detail
def _slack_presentation__task_field_list(row: slack_presentation.TaskRow):
    """Compatibility helper for labeled requested/existing comparisons."""
    fields = [('Task', row.name or 'Unnamed task'), ('Assignee', row.assignee), ('Due', slack_presentation.compact_date(row.due_date) if row.show_due else None), ('Priority', slack_presentation.compact_priority(row.priority)), ('Status', row.status)]
    return '\n'.join((f'• {label}: {slack_presentation.text(value)}' for label, value in fields if value is not None))
slack_presentation.task_field_list = _slack_presentation__task_field_list
def _slack_presentation__created_collection(rows, list_name='Action Items', today=None):
    rows = list(rows)
    count = len(rows)
    header = '*✓ Action Item Created*' if count == 1 else f'*✓ Action Items Created* · {count}'
    if count == 1:
        row = rows[0]
        details = [slack_presentation.compact_priority(row.priority) or '—', row.assignee or 'Unassigned']
        if row.show_due:
            details.append(slack_presentation.compact_date(row.due_date, today))
        details.append(row.status or 'Pending')
        body = f'*{slack_presentation.text(row.name or 'Unnamed task')}*\n\n' + ' · '.join((slack_presentation.text(value) for value in details if value is not None))
    else:
        body = '\n'.join((slack_presentation.task_line(row, position=index if count > 5 else None, bullet=count <= 5, show_status=False, today=today) for index, row in enumerate(rows, 1)))
    subject = 'Task' if count == 1 else 'Tasks'
    return header + ('\n\n' + body if body else '') + f'\n\n{subject} created successfully and verified in *{slack_presentation.text(list_name)}*.'
slack_presentation.created_collection = _slack_presentation__created_collection
def _slack_presentation__task_conflict(requested: slack_presentation.TaskRow, existing: slack_presentation.TaskRow, today=None):
    """Compact duplicate-field conflict that never mutates either row."""
    owner = existing.assignee or requested.assignee
    identity = f'*{slack_presentation.text(requested.name or existing.name or 'Unnamed task')}*'
    if owner:
        identity += f' · {slack_presentation.text(owner)}'

    def values(row):
        priority = slack_presentation.compact_priority(row.priority)
        due = slack_presentation.compact_date(row.due_date, today) if row.show_due else None
        return ' · '.join((slack_presentation.text(value) for value in (priority, due) if value is not None))
    return '*↔ Existing task differs*\n\n' + identity + f'\nRequested: {values(requested)}' + f'\nExisting: {values(existing)}' + '\n\nNo changes made — the existing task has different fields.'
slack_presentation.task_conflict = _slack_presentation__task_conflict
def _slack_presentation__assignee_clarification(row: slack_presentation.TaskRow, spoken_name, today=None):
    summary = slack_presentation.task_summary(row, show_assignee=False, show_priority=False, show_status=False, today=today)
    return f'''*⚠ Assignee unclear*\n\n{summary}\n\nI couldn't confidently match "{slack_presentation.text(spoken_name)}" to a Slack member.\nPlease mention the person or provide their Slack name.'''
slack_presentation.assignee_clarification = _slack_presentation__assignee_clarification
def _slack_presentation__bar(value, maximum, width=8):
    filled = 0 if maximum <= 0 else round(width * value / maximum)
    return '█' * filled + '░' * (width - filled)
slack_presentation.bar = _slack_presentation__bar
def _slack_presentation__distribution(title, values):
    if not values:
        return f'*{slack_presentation.text(title)}* · No reliable data available.'
    return slack_presentation.render_slack_table(('Metric', 'Count'), tuple(((label, count) for label, count in values.items())), title=title)
slack_presentation.distribution = _slack_presentation__distribution
def _slack_presentation__series(title, value):
    values = (value or {}).get('values') or {}
    if not values:
        return f'*{slack_presentation.text(title)}* · No reliable timestamped records are available.'
    return slack_presentation.render_slack_table(('Period', 'Count'), tuple(((label, count) for label, count in values.items())), title=title)
slack_presentation.series = _slack_presentation__series


# visualization.py
'Presentation-only renderers for structured project analytics.\n\nThis module contains no Slack retrieval, filtering, RBAC, or calculations.\nIts text renderer can later sit beside Block Kit or web renderers without\nchanging the analytics engine.\n'
from typing import Callable as _visualization__Callable, Iterable as _visualization__Iterable
visualization.Callable = _visualization__Callable
visualization.Iterable = _visualization__Iterable
visualization.slack_presentation = slack_presentation
def _visualization__bar(value, maximum, width=8):
    return slack_presentation.bar(value, maximum, width)
visualization.bar = _visualization__bar
def _visualization__distribution(title, values):
    return slack_presentation.distribution(title, values)
visualization.distribution = _visualization__distribution
def _visualization__series(title, value):
    return slack_presentation.series(title, value)
visualization.series = _visualization__series
def _visualization__render_progress(report, format_items: _visualization__Callable[[_visualization__Iterable, str], str]):
    """Render a ProgressReport-like value as Slack mrkdwn."""
    requested = set(report.requested)
    if 'summary' in requested:
        requested.update({'overview', 'workload', 'at_risk', 'upcoming', 'completed_over_time'})
    sections = []
    snapshot = report.snapshot
    if requested & {'overview', 'completion'}:
        rate = snapshot.get('completion_rate')
        progress = 'Unavailable' if rate is None else f'{visualization.bar(rate, 100)} {rate:g}%'
        sections.append(f'*Progress* · {snapshot['total']} total · {snapshot['completed']} completed · {snapshot['pending']} pending · {snapshot['overdue']} overdue\n• Completion {progress} · Due today {snapshot['due_today']} · Due this week {snapshot['due_this_week']}')
    if 'workload' in requested:
        pending = {name: values['pending'] for name, values in report.workload.items()}
        sections.append(visualization.distribution('Pending workload by assignee', pending))
    if 'status_distribution' in requested:
        sections.append(visualization.distribution('Status distribution', report.status_distribution))
    if 'priority_distribution' in requested:
        sections.append(visualization.distribution('Priority distribution', report.priority_distribution))
    if 'completed_over_time' in requested:
        sections.append(visualization.series('Completed tasks over time', report.completed_series))
    if 'created_over_time' in requested:
        sections.append(visualization.series('Created tasks over time', report.created_series))
    if 'comparison' in requested:
        if report.comparison.get('available'):
            sections.append(visualization.distribution('Completed-task comparison', {'This period': report.comparison['current'], 'Previous period': report.comparison['previous']}))
        else:
            sections.append('*Completed-task comparison*\nNo reliable timestamped records are available.')
    item_metrics = (('overdue', 'Overdue tasks', report.overdue_items), ('due_today', 'Tasks due today', report.due_today_items), ('due_this_week', 'Tasks due this week', report.due_this_week_items), ('upcoming', 'Upcoming deadlines', report.upcoming_items[:10]), ('at_risk', 'At-risk tasks', report.at_risk_items[:10]))
    for metric, title, items in item_metrics:
        if metric not in requested:
            continue
        sections.append(format_items(items, title))
        report.displayed_items.extend(items)
    if report.limitations:
        sections.append('*Data limitations*\n' + '\n'.join((f'• {message}' for message in dict.fromkeys(report.limitations))))
    return slack_presentation.join_sections(*sections) if sections else 'No progress metric was requested.'
visualization.render_progress = _visualization__render_progress


# progress_engine.py
'Deterministic progress calculations over Slack List item snapshots.\n\nThis module never calls Slack and never interprets language. Callers provide\nalready-authorized List records, schema metadata, requested metrics and dates.\n'
from collections import Counter as _progress_engine__Counter, defaultdict as _progress_engine__defaultdict
progress_engine.Counter = _progress_engine__Counter
progress_engine.defaultdict = _progress_engine__defaultdict
from dataclasses import dataclass as _progress_engine__dataclass, field as _progress_engine__field
progress_engine.dataclass = _progress_engine__dataclass
progress_engine.field = _progress_engine__field
from datetime import date as _progress_engine__date, datetime as _progress_engine__datetime, timedelta as _progress_engine__timedelta, timezone as _progress_engine__timezone
progress_engine.date = _progress_engine__date
progress_engine.datetime = _progress_engine__datetime
progress_engine.timedelta = _progress_engine__timedelta
progress_engine.timezone = _progress_engine__timezone
from typing import Callable as _progress_engine__Callable, Iterable as _progress_engine__Iterable
progress_engine.Callable = _progress_engine__Callable
progress_engine.Iterable = _progress_engine__Iterable
from src import slack_client as _progress_engine__slack_tools
progress_engine.slack_tools = _progress_engine__slack_tools
progress_engine.visualization = visualization
_progress_engine__METRICS = {'overview', 'completion', 'workload', 'status_distribution', 'priority_distribution', 'overdue', 'due_today', 'due_this_week', 'upcoming', 'at_risk', 'completed_over_time', 'created_over_time', 'comparison', 'summary'}
progress_engine.METRICS = _progress_engine__METRICS
@_progress_engine__dataclass
class _progress_engine__ProgressReport:
    requested: tuple[str, ...]
    snapshot: dict = _progress_engine__field(default_factory=dict)
    workload: dict = _progress_engine__field(default_factory=dict)
    status_distribution: dict = _progress_engine__field(default_factory=dict)
    priority_distribution: dict = _progress_engine__field(default_factory=dict)
    overdue_workload: dict = _progress_engine__field(default_factory=dict)
    overdue_items: list = _progress_engine__field(default_factory=list)
    due_today_items: list = _progress_engine__field(default_factory=list)
    due_this_week_items: list = _progress_engine__field(default_factory=list)
    upcoming_items: list = _progress_engine__field(default_factory=list)
    at_risk_items: list = _progress_engine__field(default_factory=list)
    completed_series: dict = _progress_engine__field(default_factory=dict)
    created_series: dict = _progress_engine__field(default_factory=dict)
    comparison: dict = _progress_engine__field(default_factory=dict)
    limitations: list[str] = _progress_engine__field(default_factory=list)
    displayed_items: list = _progress_engine__field(default_factory=list)
progress_engine.ProgressReport = _progress_engine__ProgressReport
def _progress_engine___parse_date(value):
    """Return a date only when Slack supplied a recognizable date/timestamp."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)):
        for part in value:
            parsed = progress_engine._parse_date(part)
            if parsed:
                return parsed
        return None
    if isinstance(value, dict):
        for key in ('timestamp', 'date', 'datetime', 'value', 'text'):
            parsed = progress_engine._parse_date(value.get(key))
            if parsed:
                return parsed
        return None
    if isinstance(value, (int, float)) or str(value).strip().replace('.', '', 1).isdigit():
        try:
            raw = float(value)
            if raw > 10000000000:
                raw /= 1000
            return _progress_engine__datetime.fromtimestamp(raw, ZoneInfo('Asia/Kathmandu')).date()
        except (ValueError, TypeError, OSError, OverflowError):
            return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return _progress_engine__datetime.fromisoformat(raw.replace('Z', '+00:00')).date()
    except ValueError:
        try:
            return _progress_engine__date.fromisoformat(raw[:10])
        except ValueError:
            return None
progress_engine._parse_date = _progress_engine___parse_date
def _progress_engine___timestamp_from_schema(item, schema, dimension):
    for field in _progress_engine__slack_tools._schema_fields(schema):
        label = ' '.join((str(field.get(key) or '') for key in ('key', 'name'))).casefold()
        if dimension not in label or not any((word in label for word in ('date', 'time', ' at'))):
            continue
        parsed = progress_engine._parse_date(_progress_engine__slack_tools.extract_field_value(item, schema, field.get('key') or field.get('name')))
        if parsed:
            return parsed
    return None
progress_engine._timestamp_from_schema = _progress_engine___timestamp_from_schema
def _progress_engine__completion_date(item, schema):
    for key in ('completed_at', 'completion_timestamp', 'completed_timestamp', 'date_completed'):
        parsed = progress_engine._parse_date(item.get(key))
        if parsed:
            return parsed
    return progress_engine._timestamp_from_schema(item, schema, 'complet')
progress_engine.completion_date = _progress_engine__completion_date
def _progress_engine__creation_date(item, schema):
    for key in ('date_created', 'created_timestamp', 'created_at', 'creation_timestamp'):
        parsed = progress_engine._parse_date(item.get(key))
        if parsed:
            return parsed
    return progress_engine._timestamp_from_schema(item, schema, 'creat')
progress_engine.creation_date = _progress_engine__creation_date
def _progress_engine___is_completed(item, schema):
    completed_column = _progress_engine__slack_tools.column(schema, keys=_progress_engine__slack_tools.COMPLETED_KEYS, names={'Completed'}, types={'todo_completed', 'completed', 'checkbox'})
    if completed_column:
        return _progress_engine__slack_tools.extract_completed(item, schema)
    return _progress_engine__slack_tools.extract_status(item, schema) == 'completed'
progress_engine._is_completed = _progress_engine___is_completed
def _progress_engine___due(item, schema):
    return progress_engine._parse_date(_progress_engine__slack_tools.extract_due_date(item, schema))
progress_engine._due = _progress_engine___due
def _progress_engine__calculate_completion_rate(tasks, schema):
    total = len(tasks)
    completed = sum((1 for item in tasks if progress_engine._is_completed(item, schema)))
    return round(completed * 100 / total, 1) if total else 0.0
progress_engine.calculate_completion_rate = _progress_engine__calculate_completion_rate
def _progress_engine__calculate_workload(tasks, schema, name_for_user: _progress_engine__Callable[[str], str], today=None):
    workload = _progress_engine__defaultdict(lambda: {'total': 0, 'pending': 0, 'completed': 0, 'overdue': 0})
    today = today or _progress_engine__date.today()
    for item in tasks:
        owners = _progress_engine__slack_tools.extract_assignee_ids(item, schema) or [None]
        done = progress_engine._is_completed(item, schema)
        overdue = bool(not done and progress_engine._due(item, schema) and (progress_engine._due(item, schema) < today))
        for owner in owners:
            label = name_for_user(owner) if owner else 'Unassigned'
            row = workload[label]
            row['total'] += 1
            row['completed' if done else 'pending'] += 1
            row['overdue'] += int(overdue)
    return dict(sorted(workload.items(), key=lambda pair: (-pair[1]['pending'], pair[0].casefold())))
progress_engine.calculate_workload = _progress_engine__calculate_workload
def _progress_engine__calculate_priority_distribution(tasks, schema):
    values = _progress_engine__Counter((_progress_engine__slack_tools.extract_priority(item, schema) or 'Unspecified' for item in tasks))
    order = ('P1', 'P2', 'P3', 'P4', 'Unspecified')
    return {key: values[key] for key in order if values[key]}
progress_engine.calculate_priority_distribution = _progress_engine__calculate_priority_distribution
def _progress_engine__calculate_status_distribution(tasks, schema, today=None, include_overdue=True, requested_statuses=None):
    """Return mutually exclusive completion, pending and overdue buckets."""
    today = today or _progress_engine__date.today()
    values = _progress_engine__Counter()
    for item in tasks:
        if progress_engine._is_completed(item, schema):
            values['Completed'] += 1
        elif include_overdue and progress_engine._due(item, schema) and (progress_engine._due(item, schema) < today):
            values['Overdue'] += 1
        else:
            values['Pending'] += 1
    if requested_statuses:
        binary = _progress_engine__Counter()
        for item in tasks:
            binary['Completed' if progress_engine._is_completed(item, schema) else 'Pending'] += 1
        labels = ['Completed' if value == 'completed' else 'Pending' for value in requested_statuses]
        return {label: binary[label] for label in dict.fromkeys(labels)}
    return {key: values[key] for key in ('Completed', 'Pending', 'Overdue') if values[key]}
progress_engine.calculate_status_distribution = _progress_engine__calculate_status_distribution
def _progress_engine__calculate_time_series(tasks, schema, dimension, period=None):
    extractor = progress_engine.completion_date if dimension == 'completed' else progress_engine.creation_date
    counts = _progress_engine__Counter()
    available = 0
    missing = 0
    start = progress_engine._parse_date((period or {}).get('start'))
    end = progress_engine._parse_date((period or {}).get('end'))
    for item in tasks:
        if dimension == 'completed' and (not progress_engine._is_completed(item, schema)):
            continue
        observed = extractor(item, schema)
        if not observed:
            missing += 1
            continue
        available += 1
        if start and observed < start or (end and observed > end):
            continue
        counts[observed.isoformat()] += 1
    return {'values': dict(sorted(counts.items())), 'available': available, 'missing': missing}
progress_engine.calculate_time_series = _progress_engine__calculate_time_series
def _progress_engine__calculate_progress(tasks, schema, today=None, name_for_user=None, available_fields=None, metrics=None, period=None, comparison_periods=None, requested_statuses=None):
    """Calculate requested metrics from an authorized, current List snapshot."""
    tasks = list(tasks)
    today = today or _progress_engine__date.today()
    name_for_user = name_for_user or (lambda user_id: user_id)
    if available_fields is None:
        available_fields = {'status', 'due_date', 'priority', 'assignee'}
    requested = tuple(dict.fromkeys(metrics or ('overview',)))
    requested_set = set(requested)
    wants_summary = 'summary' in requested_set
    report = progress_engine.ProgressReport(requested=requested)
    completed = [item for item in tasks if progress_engine._is_completed(item, schema)] if 'status' in available_fields else []
    pending = [item for item in tasks if not progress_engine._is_completed(item, schema)] if 'status' in available_fields else []
    dated = [(item, progress_engine._due(item, schema)) for item in tasks] if 'due_date' in available_fields else []
    overdue = [item for item, due in dated if due and due < today and (item in pending)]
    due_today = [item for item, due in dated if due == today and item in pending]
    week_end = today + _progress_engine__timedelta(days=6 - today.weekday())
    due_week = [item for item, due in dated if due and today <= due <= week_end and (item in pending)]
    report.snapshot = {'total': len(tasks), 'completed': len(completed), 'pending': len(pending), 'overdue': len(overdue), 'due_today': len(due_today), 'due_this_week': len(due_week), 'completion_rate': progress_engine.calculate_completion_rate(tasks, schema) if 'status' in available_fields else None}
    if 'status' not in available_fields and (wants_summary or requested_set & {'overview', 'completion', 'workload', 'status_distribution', 'at_risk', 'completed_over_time'}):
        report.limitations.append('Task completion/status is unavailable or not readable for this List.')
    if 'due_date' not in available_fields and (wants_summary or requested_set & {'overview', 'overdue', 'due_today', 'due_this_week', 'upcoming', 'at_risk'}):
        report.limitations.append('Due-date metrics are unavailable or not readable for this List.')
    wants_workload = wants_summary or 'workload' in requested_set
    if wants_workload and {'assignee', 'status'}.issubset(available_fields):
        report.workload = progress_engine.calculate_workload(tasks, schema, name_for_user, today)
        report.overdue_workload = {name: values['overdue'] for name, values in report.workload.items() if values['overdue']}
    elif wants_workload and 'assignee' not in available_fields:
        report.limitations.append('Assignee workload is unavailable or not readable for this List.')
    elif wants_workload:
        report.limitations.append('Pending workload requires a readable completion/status field.')
    if 'priority_distribution' in requested_set and 'priority' in available_fields:
        report.priority_distribution = progress_engine.calculate_priority_distribution(tasks, schema)
    elif 'priority_distribution' in requested_set:
        report.limitations.append('Priority distribution is unavailable or not readable for this List.')
    if 'status_distribution' in requested_set and 'status' in available_fields:
        report.status_distribution = progress_engine.calculate_status_distribution(tasks, schema, today, include_overdue='due_date' in available_fields, requested_statuses=requested_statuses)
    report.overdue_items = overdue
    report.due_today_items = due_today
    report.due_this_week_items = due_week
    report.upcoming_items = [item for item, due in sorted(dated, key=lambda pair: pair[1] or _progress_engine__date.max) if due and due >= today and (item in pending)]
    if wants_summary or 'at_risk' in requested_set:
        for item in pending:
            due = progress_engine._due(item, schema) if 'due_date' in available_fields else None
            priority = _progress_engine__slack_tools.extract_priority(item, schema) if 'priority' in available_fields else None
            if due and due <= today + _progress_engine__timedelta(days=3) or priority == 'P1':
                report.at_risk_items.append(item)
    if (wants_summary or 'at_risk' in requested_set) and ('status' not in available_fields or not {'due_date', 'priority'} & set(available_fields)):
        report.limitations.append('At-risk tasks require readable status and due-date or priority fields.')
    if 'completed_over_time' in requested_set or wants_summary:
        report.completed_series = progress_engine.calculate_time_series(tasks, schema, 'completed', period) if 'status' in available_fields else {'values': {}, 'available': 0, 'missing': 0}
        if not report.completed_series['available']:
            report.limitations.append('Slack List does not expose reliable completion timestamps for these tasks, so completed work over time cannot be calculated.')
    if 'created_over_time' in requested_set:
        report.created_series = progress_engine.calculate_time_series(tasks, schema, 'created', period)
        if not report.created_series['available']:
            report.limitations.append('Slack List does not expose reliable creation timestamps for these tasks, so created work over time cannot be calculated.')
    if 'comparison' in requested_set:
        current_series = progress_engine.calculate_time_series(tasks, schema, 'completed', (comparison_periods or {}).get('current'))
        previous_series = progress_engine.calculate_time_series(tasks, schema, 'completed', (comparison_periods or {}).get('previous'))
        report.comparison = {'current': sum(current_series['values'].values()), 'previous': sum(previous_series['values'].values()), 'available': max(current_series['available'], previous_series['available'])}
        if not report.comparison['available']:
            report.limitations.append('Slack List does not expose reliable completion timestamps, so period comparison cannot be calculated.')
    return report
progress_engine.calculate_progress = _progress_engine__calculate_progress
def _progress_engine__render_progress(report: progress_engine.ProgressReport, format_items: _progress_engine__Callable[[_progress_engine__Iterable, str], str]):
    """Backward-compatible entry point for the modular Slack renderer."""
    return visualization.render_progress(report, format_items)
progress_engine.render_progress = _progress_engine__render_progress


# project_intelligence.py
'Deterministic project-management insights over authorized Slack List data.\n\nThe functions in this module are deliberately unaware of Slack transport,\nconversation state, RBAC and language parsing.  Callers supply current records\nand schema metadata; this module only calculates explainable results.\n'
from dataclasses import dataclass as _project_intelligence__dataclass, field as _project_intelligence__field
project_intelligence.dataclass = _project_intelligence__dataclass
project_intelligence.field = _project_intelligence__field
from datetime import date as _project_intelligence__date, timedelta as _project_intelligence__timedelta
project_intelligence.date = _project_intelligence__date
project_intelligence.timedelta = _project_intelligence__timedelta
from statistics import mean as _project_intelligence__mean
project_intelligence.mean = _project_intelligence__mean
from typing import Callable as _project_intelligence__Callable, Iterable as _project_intelligence__Iterable
project_intelligence.Callable = _project_intelligence__Callable
project_intelligence.Iterable = _project_intelligence__Iterable
from src import slack_client as _project_intelligence__slack_tools
project_intelligence.slack_tools = _project_intelligence__slack_tools
_project_intelligence__PRIORITY_RANK = {'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4}
project_intelligence.PRIORITY_RANK = _project_intelligence__PRIORITY_RANK
@_project_intelligence__dataclass(frozen=True)
class _project_intelligence__TaskHealth:
    item_id: str
    item: dict
    level: str
    icon: str
    reasons: tuple[str, ...]
project_intelligence.TaskHealth = _project_intelligence__TaskHealth
@_project_intelligence__dataclass(frozen=True)
class _project_intelligence__PlanEntry:
    item_id: str
    item: dict
    scheduled_date: _project_intelligence__date
    reasons: tuple[str, ...]
project_intelligence.PlanEntry = _project_intelligence__PlanEntry
@_project_intelligence__dataclass
class _project_intelligence__WorkloadReport:
    rows: dict = _project_intelligence__field(default_factory=dict)
    overloaded: list[str] = _project_intelligence__field(default_factory=list)
    suggestions: list[dict] = _project_intelligence__field(default_factory=list)
project_intelligence.WorkloadReport = _project_intelligence__WorkloadReport
@_project_intelligence__dataclass
class _project_intelligence__StandupReport:
    completed: list = _project_intelligence__field(default_factory=list)
    pending: list = _project_intelligence__field(default_factory=list)
    attention: list[project_intelligence.TaskHealth] = _project_intelligence__field(default_factory=list)
    upcoming: list = _project_intelligence__field(default_factory=list)
    completion_is_daily: bool = False
    limitations: list[str] = _project_intelligence__field(default_factory=list)
project_intelligence.StandupReport = _project_intelligence__StandupReport
@_project_intelligence__dataclass(frozen=True)
class _project_intelligence__FocusEntry:
    item_id: str
    item: dict
    due_date: _project_intelligence__date | None
    priority: str | None
    reason: str
    category: str
project_intelligence.FocusEntry = _project_intelligence__FocusEntry
@_project_intelligence__dataclass(frozen=True)
class _project_intelligence__RiskEntry:
    item_id: str
    item: dict
    reasons: tuple[str, ...]
project_intelligence.RiskEntry = _project_intelligence__RiskEntry
@_project_intelligence__dataclass(frozen=True)
class _project_intelligence__HealthSummary:
    pending: int
    completed: int
    overdue: int
    p1: int
    unassigned: int
    due_within_48h: int
    risks: tuple[project_intelligence.RiskEntry, ...]
project_intelligence.HealthSummary = _project_intelligence__HealthSummary
@_project_intelligence__dataclass(frozen=True)
class _project_intelligence__NormalizedTask:
    """One immutable intelligence view over a current Slack List item."""
    item_id: str
    item: dict
    name: str
    owner_ids: tuple[str, ...]
    priority: str | None
    due_date: _project_intelligence__date | None
    completed: bool
    status: str
    created_date: _project_intelligence__date | None
project_intelligence.NormalizedTask = _project_intelligence__NormalizedTask
def _project_intelligence___as_date(value):
    if not value:
        return None
    try:
        return _project_intelligence__date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None
project_intelligence._as_date = _project_intelligence___as_date
def _project_intelligence___schema_fields(schema):
    if isinstance(schema, list):
        return schema
    if not isinstance(schema, dict):
        return []
    return schema.get('schema') or schema.get('fields') or schema.get('columns') or []
project_intelligence._schema_fields = _project_intelligence___schema_fields
def _project_intelligence__normalize_task(item, schema):
    """Normalize one raw Slack List item through the existing field extractors."""
    if isinstance(item, project_intelligence.NormalizedTask):
        return item
    status = str(_project_intelligence__slack_tools.extract_status(item, schema) or '').strip()
    priority = _project_intelligence__slack_tools.normalize_priority(_project_intelligence__slack_tools.extract_priority(item, schema), schema)
    return project_intelligence.NormalizedTask(item_id=_project_intelligence__slack_tools.extract_item_id(item), item=item, name=_project_intelligence__slack_tools.extract_item_name(item, schema) or 'Unnamed task', owner_ids=tuple(_project_intelligence__slack_tools.extract_assignee_ids(item, schema)), priority=priority, due_date=project_intelligence._as_date(_project_intelligence__slack_tools.extract_due_date(item, schema)), completed=_project_intelligence__slack_tools.extract_completed(item, schema) or status.casefold() in {'cancelled', 'canceled', 'closed'}, status=status, created_date=progress_engine.creation_date(item, schema))
project_intelligence.normalize_task = _project_intelligence__normalize_task
def _project_intelligence__normalize_task_snapshot(tasks, schema):
    """Create the single normalized snapshot consumed by intelligence features."""
    values = list(tasks)
    if values and all((isinstance(item, project_intelligence.NormalizedTask) for item in values)):
        return values
    return [project_intelligence.normalize_task(item, schema) for item in values]
project_intelligence.normalize_task_snapshot = _project_intelligence__normalize_task_snapshot
def _project_intelligence__calculate_daily_focus(tasks, schema, today=None):
    """Rank pending work using factual deadline and priority dimensions."""
    today = today or _project_intelligence__date.today()
    entries = []
    for task in project_intelligence.normalize_task_snapshot(tasks, schema):
        if task.completed:
            continue
        due = task.due_date
        priority = task.priority
        if due and due < today:
            days = (today - due).days
            reason = f'{(priority + ' + ' if priority == 'P1' else '')}overdue by {days} day{('s' if days != 1 else '')}'
            deadline_rank = 0
            category = 'immediate'
        elif due == today:
            reason = 'high priority + due today' if priority == 'P1' else 'due today'
            deadline_rank = 1
            category = 'immediate'
        elif due == today + _project_intelligence__timedelta(days=1):
            reason = 'high priority + approaching deadline' if priority == 'P1' else 'due tomorrow'
            deadline_rank = 2
            category = 'upcoming'
        elif due:
            reason = 'upcoming deadline'
            deadline_rank = 3
            category = 'upcoming'
        else:
            reason = 'pending with no due date'
            deadline_rank = 4
            category = 'upcoming'
        entries.append((deadline_rank, project_intelligence.PRIORITY_RANK.get(priority, 5), due or _project_intelligence__date.max, task.name.casefold(), project_intelligence.FocusEntry(task.item_id, task.item, due, priority, reason, category)))
    return [entry[-1] for entry in sorted(entries, key=lambda value: value[:-1])]
project_intelligence.calculate_daily_focus = _project_intelligence__calculate_daily_focus
def _project_intelligence__analyze_task_risks(tasks, schema, today=None):
    """Return explainable risk signals without assigning a subjective score."""
    today = today or _project_intelligence__date.today()
    assignee_available = bool(_project_intelligence__slack_tools.column(schema, keys=_project_intelligence__slack_tools.ASSIGNEE_KEYS, names={'Assignee', 'Owner'}))
    risks = []
    normalized = project_intelligence.normalize_task_snapshot(tasks, schema)
    for task in normalized:
        if task.completed:
            continue
        due = task.due_date
        priority = task.priority
        owners = task.owner_ids
        reasons = []
        if due and due < today:
            days = (today - due).days
            reasons.append(f'Overdue by {days} day{('s' if days != 1 else '')}')
        elif due == today:
            reasons.append('Due today')
        elif due and due <= today + _project_intelligence__timedelta(days=2):
            hours = (due - today).days * 24
            reasons.append(f'Due within {hours} hours')
        if priority == 'P1':
            reasons.append('P1 priority')
        if assignee_available and (not owners):
            reasons.append('No assigned owner')
        if reasons:
            risks.append(project_intelligence.RiskEntry(task.item_id, task.item, tuple(reasons)))
    by_id = {task.item_id: task for task in normalized}

    def rank(record):
        task = by_id[record.item_id]
        due = task.due_date
        return (0 if due and due < today else 1, 0 if task.priority == 'P1' else 1, due or _project_intelligence__date.max, task.name.casefold())
    return sorted(risks, key=rank)
project_intelligence.analyze_task_risks = _project_intelligence__analyze_task_risks
def _project_intelligence__generate_task_health(tasks, schema, today=None):
    """Calculate aggregate health facts from one current task snapshot."""
    today = today or _project_intelligence__date.today()
    normalized = project_intelligence.normalize_task_snapshot(tasks, schema)
    completed = [task for task in normalized if task.completed]
    pending = [task for task in normalized if not task.completed]
    overdue = 0
    due_within_48h = 0
    for task in pending:
        due = task.due_date
        if due and due < today:
            overdue += 1
        elif due and today <= due <= today + _project_intelligence__timedelta(days=2):
            due_within_48h += 1
    risks = project_intelligence.analyze_task_risks(pending, schema, today)
    assignee_available = bool(_project_intelligence__slack_tools.column(schema, keys=_project_intelligence__slack_tools.ASSIGNEE_KEYS, names={'Assignee', 'Owner'}))
    return project_intelligence.HealthSummary(pending=len(pending), completed=len(completed), overdue=overdue, p1=sum((1 for task in pending if task.priority == 'P1')), unassigned=sum((1 for task in pending if not task.owner_ids)) if assignee_available else 0, due_within_48h=due_within_48h, risks=tuple(risks))
project_intelligence.generate_task_health = _project_intelligence__generate_task_health
def _project_intelligence__generate_weekly_insights(tasks, schema, name_for_user, today=None):
    """Build concise, factual insight sentences from pending tasks."""
    today = today or _project_intelligence__date.today()
    normalized = project_intelligence.normalize_task_snapshot(tasks, schema)
    health = project_intelligence.generate_task_health(normalized, schema, today)
    insights = []
    if health.overdue:
        insights.append(f'{health.overdue} task{('s are' if health.overdue != 1 else ' is')} overdue.')
    if health.due_within_48h:
        insights.append(f'{health.due_within_48h} task{('s are' if health.due_within_48h != 1 else ' is')} due within 48 hours.')
    if health.unassigned:
        insights.append(f'{health.unassigned} pending task{('s are' if health.unassigned != 1 else ' is')} unassigned.')
    p1_by_owner = {}
    for task in normalized:
        if task.completed or task.priority != 'P1':
            continue
        for owner_id in task.owner_ids:
            p1_by_owner[owner_id] = p1_by_owner.get(owner_id, 0) + 1
    for owner_id, count in sorted(p1_by_owner.items(), key=lambda pair: (-pair[1], name_for_user(pair[0]).casefold())):
        if count >= 2:
            insights.append(f'{name_for_user(owner_id)} has {count} pending P1 tasks.')
    return insights
project_intelligence.generate_weekly_insights = _project_intelligence__generate_weekly_insights
def _project_intelligence__dependency_fields(schema):
    """Discover explicit dependency/blocking fields without inventing schema."""
    discovered = []
    for field_value in project_intelligence._schema_fields(schema):
        label = ' '.join((str(field_value.get(key) or '') for key in ('key', 'name', 'title'))).casefold()
        if any((token in label for token in ('depend', 'blocked by', 'blocker', 'blocking'))):
            discovered.append(field_value)
    return discovered
project_intelligence.dependency_fields = _project_intelligence__dependency_fields
def _project_intelligence__dependency_values(item, schema):
    values = []
    for schema_field in project_intelligence.dependency_fields(schema):
        key = schema_field.get('key') or schema_field.get('name') or schema_field.get('id')
        value = _project_intelligence__slack_tools.extract_field_value(item, schema, key)
        if value not in (None, '', [], False):
            values.append(str(value))
    return tuple(values)
project_intelligence.dependency_values = _project_intelligence__dependency_values
def _project_intelligence__classify_task(item, schema, today=None, attention_days=3):
    """Classify one task from factual fields and retain every reason."""
    today = today or _project_intelligence__date.today()
    task = project_intelligence.normalize_task(item, schema)
    item_id = task.item_id
    completed = task.completed
    due = task.due_date
    priority = task.priority
    dependencies = project_intelligence.dependency_values(task.item, schema)
    reasons = []
    if completed:
        return project_intelligence.TaskHealth(item_id, task.item, 'On Track', '🟢', ('Completed',))
    if due and due < today:
        days = (today - due).days
        reasons.append(f'Overdue by {days} day{('s' if days != 1 else '')}')
        if priority:
            reasons.append(priority)
        if dependencies:
            reasons.append('Explicit dependency/blocker data is present')
        return project_intelligence.TaskHealth(item_id, task.item, 'Overdue', '🔴', tuple(reasons))
    if due is None:
        reasons.append('No due date')
        if priority:
            reasons.append(priority)
        if dependencies:
            reasons.append('Explicit dependency/blocker data is present')
        return project_intelligence.TaskHealth(item_id, task.item, 'No Deadline', '⚪', tuple(reasons))
    days = (due - today).days
    if days == 0:
        reasons.append('Due today')
    elif days == 1:
        reasons.append('Due tomorrow')
    elif days <= attention_days:
        reasons.append(f'Due in {days} days')
    if priority == 'P1':
        reasons.append('P1 priority')
    if dependencies:
        reasons.append('Explicit dependency/blocker data is present')
    if reasons:
        reasons.append('Still pending')
        return project_intelligence.TaskHealth(item_id, task.item, 'Needs Attention', '🟡', tuple(reasons))
    return project_intelligence.TaskHealth(item_id, task.item, 'On Track', '🟢', (f'Due {due.isoformat()}', priority or 'No priority'))
project_intelligence.classify_task = _project_intelligence__classify_task
def _project_intelligence__calculate_health(tasks, schema, today=None, attention_only=False, attention_days=3):
    snapshot = project_intelligence.normalize_task_snapshot(tasks, schema)
    records = [project_intelligence.classify_task(task, schema, today, attention_days) for task in snapshot]
    if attention_only:
        records = [record for record in records if record.level in {'Needs Attention', 'Overdue'}]
    order = {'Overdue': 0, 'Needs Attention': 1, 'No Deadline': 2, 'On Track': 3}
    return sorted(records, key=lambda record: (order[record.level], _project_intelligence__slack_tools.extract_item_name(record.item, schema).casefold()))
project_intelligence.calculate_health = _project_intelligence__calculate_health
def _project_intelligence___weekdays(start, end):
    days = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += _project_intelligence__timedelta(days=1)
    return days
project_intelligence._weekdays = _project_intelligence___weekdays
def _project_intelligence__build_plan(tasks, schema, start, end, today=None):
    """Create a due-date proposal; never mutates records."""
    today = today or _project_intelligence__date.today()
    days = project_intelligence._weekdays(start, end)
    if not days:
        raise ValueError('The planning period does not contain a working day.')
    pending = [item for item in tasks if not _project_intelligence__slack_tools.extract_completed(item, schema)]

    def rank(item):
        due = project_intelligence._as_date(_project_intelligence__slack_tools.extract_due_date(item, schema))
        return (0 if due and due < today else 1, due or _project_intelligence__date.max, project_intelligence.PRIORITY_RANK.get(_project_intelligence__slack_tools.extract_priority(item, schema), 5), _project_intelligence__slack_tools.extract_item_name(item, schema).casefold())
    entries = []
    for index, item in enumerate(sorted(pending, key=rank)):
        original_due = project_intelligence._as_date(_project_intelligence__slack_tools.extract_due_date(item, schema))
        scheduled = days[min(index, len(days) - 1)]
        reasons = []
        if original_due and original_due < today:
            reasons.append('currently overdue')
        elif original_due:
            reasons.append(f'current deadline {original_due.isoformat()}')
        else:
            reasons.append('no current deadline')
        priority = _project_intelligence__slack_tools.extract_priority(item, schema)
        if priority:
            reasons.append(priority)
        entries.append(project_intelligence.PlanEntry(_project_intelligence__slack_tools.extract_item_id(item), item, scheduled, tuple(reasons)))
    return entries
project_intelligence.build_plan = _project_intelligence__build_plan
def _project_intelligence__calculate_workload(tasks, schema, name_for_user: _project_intelligence__Callable[[str], str], today=None, eligible_user_ids: _project_intelligence__Iterable[str]=()):
    """Calculate workload and deterministic rebalance suggestions."""
    today = today or _project_intelligence__date.today()
    normalized = project_intelligence.normalize_task_snapshot(tasks, schema)
    by_user = {user_id: [] for user_id in eligible_user_ids}
    unassigned = []
    for task in normalized:
        if task.completed:
            continue
        owners = task.owner_ids
        if not owners:
            unassigned.append(task)
        for owner in owners:
            by_user.setdefault(owner, []).append(task)
    rows = {}
    scores = {}
    for user_id, owned in by_user.items():
        overdue = sum((1 for task in owned if (task.due_date or _project_intelligence__date.max) < today))
        p1 = sum((1 for task in owned if task.priority == 'P1'))
        p2 = sum((1 for task in owned if task.priority == 'P2'))
        p3 = sum((1 for task in owned if task.priority == 'P3'))
        due_soon = sum((1 for task in owned if task.due_date is not None and today <= task.due_date <= today + _project_intelligence__timedelta(days=2)))
        upcoming = sum((1 for task in owned if task.due_date is not None and today <= task.due_date <= today + _project_intelligence__timedelta(days=7)))
        score = len(owned) + 2 * overdue + 2 * p1
        rows[user_id] = {'name': name_for_user(user_id), 'pending': len(owned), 'overdue': overdue, 'p1': p1, 'p2': p2, 'p3': p3, 'due_soon': due_soon, 'upcoming': upcoming, 'score': score}
        scores[user_id] = score
    if unassigned:
        rows[None] = {'name': 'Unassigned', 'pending': len(unassigned), 'overdue': sum((1 for task in unassigned if (task.due_date or _project_intelligence__date.max) < today)), 'p1': sum((1 for task in unassigned if task.priority == 'P1')), 'p2': sum((1 for task in unassigned if task.priority == 'P2')), 'p3': sum((1 for task in unassigned if task.priority == 'P3')), 'due_soon': sum((1 for task in unassigned if task.due_date is not None and today <= task.due_date <= today + _project_intelligence__timedelta(days=2))), 'upcoming': sum((1 for task in unassigned if task.due_date is not None and today <= task.due_date <= today + _project_intelligence__timedelta(days=7))), 'score': len(unassigned)}
    active_scores = list(scores.values())
    average = _project_intelligence__mean(active_scores) if active_scores else 0
    overloaded = [user_id for user_id, score in scores.items() if score > average * 1.25 and score >= average + 2]
    suggestions = []
    if scores:
        recipients = sorted(scores, key=lambda user_id: (scores[user_id], name_for_user(user_id).casefold()))
        for source in sorted(overloaded, key=lambda user_id: -scores[user_id]):
            destination = next((user_id for user_id in recipients if user_id != source), None)
            if not destination or scores[source] <= scores[destination] + 2:
                continue
            movable = sorted(by_user[source], key=lambda task: (project_intelligence.PRIORITY_RANK.get(task.priority, 5), task.due_date or _project_intelligence__date.max), reverse=True)
            if movable:
                task = movable[0]
                suggestions.append({'item_id': task.item_id, 'item': task.item, 'from_user_id': source, 'to_user_id': destination, 'reason': f'workload score {scores[source]} versus {scores[destination]}'})
    return project_intelligence.WorkloadReport(rows=rows, overloaded=overloaded, suggestions=suggestions)
project_intelligence.calculate_workload = _project_intelligence__calculate_workload
def _project_intelligence__build_standup(tasks, schema, today=None):
    today = today or _project_intelligence__date.today()
    completed = [item for item in tasks if _project_intelligence__slack_tools.extract_completed(item, schema)]
    timestamped = [(item, progress_engine.completion_date(item, schema)) for item in completed]
    available = [pair for pair in timestamped if pair[1] is not None]
    report = project_intelligence.StandupReport()
    if available:
        report.completed = [item for item, completed_on in available if completed_on == today]
        report.completion_is_daily = True
        if len(available) != len(completed):
            report.limitations.append("Some completed tasks have no completion timestamp and are excluded from today's completed section.")
    else:
        report.completed = completed
        report.limitations.append('Slack List does not expose reliable completion timestamps, so completed tasks are shown as current state, not as completed today.')
    report.pending = [item for item in tasks if not _project_intelligence__slack_tools.extract_completed(item, schema)]
    report.attention = project_intelligence.calculate_health(report.pending, schema, today, attention_only=True)
    tomorrow = today + _project_intelligence__timedelta(days=1)
    report.upcoming = [item for item in report.pending if project_intelligence._as_date(_project_intelligence__slack_tools.extract_due_date(item, schema)) == tomorrow]
    return report
project_intelligence.build_standup = _project_intelligence__build_standup


# predictive_intelligence.py
'Deterministic predictive intelligence over an authorized task snapshot.\n\nThis module never reads Slack directly and never mutates task state.  It models\npressure from facts already present in ``NormalizedTask`` records and uses\ncareful language: a signal is an emerging risk, not a prediction of failure.\n'
from collections import Counter as _predictive_intelligence__Counter, defaultdict as _predictive_intelligence__defaultdict
predictive_intelligence.Counter = _predictive_intelligence__Counter
predictive_intelligence.defaultdict = _predictive_intelligence__defaultdict
from dataclasses import dataclass as _predictive_intelligence__dataclass
predictive_intelligence.dataclass = _predictive_intelligence__dataclass
from datetime import date as _predictive_intelligence__date, timedelta as _predictive_intelligence__timedelta
predictive_intelligence.date = _predictive_intelligence__date
predictive_intelligence.timedelta = _predictive_intelligence__timedelta
from typing import Callable as _predictive_intelligence__Callable, Iterable as _predictive_intelligence__Iterable
predictive_intelligence.Callable = _predictive_intelligence__Callable
predictive_intelligence.Iterable = _predictive_intelligence__Iterable
@_predictive_intelligence__dataclass(frozen=True)
class _predictive_intelligence__DeadlineCluster:
    due_date: _predictive_intelligence__date
    task_ids: tuple[str, ...]
    priority_counts: dict[str, int]
    owner_counts: dict[str | None, int]
predictive_intelligence.DeadlineCluster = _predictive_intelligence__DeadlineCluster
@_predictive_intelligence__dataclass(frozen=True)
class _predictive_intelligence__WorkloadOutlook:
    owner_id: str | None
    pending: int
    overdue: int
    p1: int
    due_within_48h: int
    due_within_7d: int
predictive_intelligence.WorkloadOutlook = _predictive_intelligence__WorkloadOutlook
@_predictive_intelligence__dataclass(frozen=True)
class _predictive_intelligence__EmergingRisk:
    task_id: str
    task_name: str
    owner_ids: tuple[str, ...]
    priority: str | None
    due_date: _predictive_intelligence__date | None
    level: str
    evidence: tuple[str, ...]
predictive_intelligence.EmergingRisk = _predictive_intelligence__EmergingRisk
@_predictive_intelligence__dataclass(frozen=True)
class _predictive_intelligence__PredictiveSummary:
    pending: int
    completed: int
    overdue: int
    due_today: int
    due_within_24h: int
    due_within_48h: int
    due_within_7d: int
    priority_counts: dict[str, int]
    unassigned: int
    workload: tuple[predictive_intelligence.WorkloadOutlook, ...]
    deadline_clusters: tuple[predictive_intelligence.DeadlineCluster, ...]
    emerging_risks: tuple[predictive_intelligence.EmergingRisk, ...]
predictive_intelligence.PredictiveSummary = _predictive_intelligence__PredictiveSummary
def _predictive_intelligence___pending(tasks: _predictive_intelligence__Iterable[project_intelligence.NormalizedTask]) -> list[project_intelligence.NormalizedTask]:
    return [task for task in tasks if not task.completed]
predictive_intelligence._pending = _predictive_intelligence___pending
def _predictive_intelligence__calculate_deadline_pressure(tasks: _predictive_intelligence__Iterable[project_intelligence.NormalizedTask], today: _predictive_intelligence__date) -> dict[str, int]:
    """Return mutually understandable deadline facts for pending work."""
    pending = predictive_intelligence._pending(tasks)
    return {'overdue': sum((bool(task.due_date and task.due_date < today) for task in pending)), 'due_today': sum((task.due_date == today for task in pending)), 'due_within_24h': sum((bool(task.due_date and today <= task.due_date <= today + _predictive_intelligence__timedelta(days=1)) for task in pending)), 'due_within_48h': sum((bool(task.due_date and today <= task.due_date <= today + _predictive_intelligence__timedelta(days=2)) for task in pending)), 'due_within_7d': sum((bool(task.due_date and today <= task.due_date <= today + _predictive_intelligence__timedelta(days=7)) for task in pending))}
predictive_intelligence.calculate_deadline_pressure = _predictive_intelligence__calculate_deadline_pressure
def _predictive_intelligence__calculate_priority_pressure(tasks: _predictive_intelligence__Iterable[project_intelligence.NormalizedTask]) -> dict[str, int]:
    pending = predictive_intelligence._pending(tasks)
    return {priority: sum((task.priority == priority for task in pending)) for priority in ('P1', 'P2', 'P3')}
predictive_intelligence.calculate_priority_pressure = _predictive_intelligence__calculate_priority_pressure
def _predictive_intelligence__calculate_workload_pressure(tasks: _predictive_intelligence__Iterable[project_intelligence.NormalizedTask], today: _predictive_intelligence__date) -> tuple[predictive_intelligence.WorkloadOutlook, ...]:
    """Calculate current and near-term owner load without subjective labels."""
    grouped: dict[str | None, list[project_intelligence.NormalizedTask]] = _predictive_intelligence__defaultdict(list)
    for task in predictive_intelligence._pending(tasks):
        if task.owner_ids:
            for owner_id in task.owner_ids:
                grouped[owner_id].append(task)
        else:
            grouped[None].append(task)
    rows = []
    for owner_id, owned in grouped.items():
        rows.append(predictive_intelligence.WorkloadOutlook(owner_id=owner_id, pending=len(owned), overdue=sum((bool(task.due_date and task.due_date < today) for task in owned)), p1=sum((task.priority == 'P1' for task in owned)), due_within_48h=sum((bool(task.due_date and today <= task.due_date <= today + _predictive_intelligence__timedelta(days=2)) for task in owned)), due_within_7d=sum((bool(task.due_date and today <= task.due_date <= today + _predictive_intelligence__timedelta(days=7)) for task in owned))))
    return tuple(sorted(rows, key=lambda row: (-row.pending, row.owner_id or '')))
predictive_intelligence.calculate_workload_pressure = _predictive_intelligence__calculate_workload_pressure
def _predictive_intelligence__calculate_deadline_concentration(tasks: _predictive_intelligence__Iterable[project_intelligence.NormalizedTask], today: _predictive_intelligence__date, *, minimum_tasks: int=2, horizon_days: int=7) -> tuple[predictive_intelligence.DeadlineCluster, ...]:
    """Find near-term dates shared by multiple pending tasks."""
    grouped: dict[_predictive_intelligence__date, list[project_intelligence.NormalizedTask]] = _predictive_intelligence__defaultdict(list)
    horizon = today + _predictive_intelligence__timedelta(days=horizon_days)
    for task in predictive_intelligence._pending(tasks):
        if task.due_date and today <= task.due_date <= horizon:
            grouped[task.due_date].append(task)
    clusters = []
    for due_date, due_tasks in grouped.items():
        if len(due_tasks) < minimum_tasks:
            continue
        owners: _predictive_intelligence__Counter[str | None] = _predictive_intelligence__Counter()
        priorities: _predictive_intelligence__Counter[str] = _predictive_intelligence__Counter()
        for task in due_tasks:
            priorities[task.priority or 'No priority'] += 1
            if task.owner_ids:
                owners.update(task.owner_ids)
            else:
                owners[None] += 1
        clusters.append(predictive_intelligence.DeadlineCluster(due_date, tuple(sorted((task.item_id for task in due_tasks))), dict(priorities), dict(owners)))
    return tuple(sorted(clusters, key=lambda cluster: cluster.due_date))
predictive_intelligence.calculate_deadline_concentration = _predictive_intelligence__calculate_deadline_concentration
def _predictive_intelligence__build_risk_evidence(task: project_intelligence.NormalizedTask, *, today: _predictive_intelligence__date, owner_outlook: predictive_intelligence.WorkloadOutlook | None, cluster: predictive_intelligence.DeadlineCluster | None) -> tuple[str, ...]:
    """Build factual evidence for one not-yet-overdue task."""
    evidence = []
    if task.due_date:
        days = (task.due_date - today).days
        if days == 0:
            evidence.append('Due today')
        elif days == 1:
            evidence.append('1 day until deadline')
        elif 1 < days <= 7:
            evidence.append(f'{days} days until deadline')
    if task.priority == 'P1':
        evidence.append('P1 priority')
    if not task.owner_ids:
        evidence.append('No assigned owner')
    if owner_outlook and owner_outlook.p1 >= 2:
        evidence.append(f'Owner has {owner_outlook.p1} pending P1 tasks')
    if owner_outlook and owner_outlook.due_within_48h >= 2:
        evidence.append(f'Owner has {owner_outlook.due_within_48h} deadlines within 48 hours')
    if cluster:
        evidence.append(f'{len(cluster.task_ids)} tasks share this deadline')
    return tuple(evidence)
predictive_intelligence.build_risk_evidence = _predictive_intelligence__build_risk_evidence
def _predictive_intelligence__detect_emerging_risks(tasks: _predictive_intelligence__Iterable[project_intelligence.NormalizedTask], today: _predictive_intelligence__date) -> tuple[predictive_intelligence.EmergingRisk, ...]:
    """Identify evidence-backed pressure before tasks become overdue."""
    values = list(tasks)
    workloads = {row.owner_id: row for row in predictive_intelligence.calculate_workload_pressure(values, today)}
    clusters = predictive_intelligence.calculate_deadline_concentration(values, today)
    cluster_by_task = {task_id: cluster for cluster in clusters for task_id in cluster.task_ids}
    risks = []
    for task in predictive_intelligence._pending(values):
        if not task.due_date or task.due_date < today or task.due_date > today + _predictive_intelligence__timedelta(days=7):
            continue
        owner = workloads.get(task.owner_ids[0]) if len(task.owner_ids) == 1 else None
        evidence = predictive_intelligence.build_risk_evidence(task, today=today, owner_outlook=owner, cluster=cluster_by_task.get(task.item_id))
        days = (task.due_date - today).days
        strong_signals = sum((task.priority == 'P1', not task.owner_ids, bool(owner and owner.p1 >= 2), task.item_id in cluster_by_task))
        qualifies = days <= 2 and strong_signals >= 1 or (days <= 7 and strong_signals >= 2)
        if not qualifies:
            continue
        level = 'high' if days <= 1 and (task.priority == 'P1' or not task.owner_ids) else 'attention'
        risks.append(predictive_intelligence.EmergingRisk(task.item_id, task.name, task.owner_ids, task.priority, task.due_date, level, evidence))
    return tuple(sorted(risks, key=lambda risk: (risk.due_date or _predictive_intelligence__date.max, 0 if risk.priority == 'P1' else 1, risk.task_name.casefold())))
predictive_intelligence.detect_emerging_risks = _predictive_intelligence__detect_emerging_risks
def _predictive_intelligence__build_predictive_summary(tasks: _predictive_intelligence__Iterable[project_intelligence.NormalizedTask], today: _predictive_intelligence__date) -> predictive_intelligence.PredictiveSummary:
    """Build one reusable summary from a single authorized snapshot."""
    values = list(tasks)
    pending = predictive_intelligence._pending(values)
    deadline = predictive_intelligence.calculate_deadline_pressure(values, today)
    return predictive_intelligence.PredictiveSummary(pending=len(pending), completed=len(values) - len(pending), overdue=deadline['overdue'], due_today=deadline['due_today'], due_within_24h=deadline['due_within_24h'], due_within_48h=deadline['due_within_48h'], due_within_7d=deadline['due_within_7d'], priority_counts=predictive_intelligence.calculate_priority_pressure(values), unassigned=sum((not task.owner_ids for task in pending)), workload=predictive_intelligence.calculate_workload_pressure(values, today), deadline_clusters=predictive_intelligence.calculate_deadline_concentration(values, today), emerging_risks=predictive_intelligence.detect_emerging_risks(values, today))
predictive_intelligence.build_predictive_summary = _predictive_intelligence__build_predictive_summary
def _predictive_intelligence__workload_forecast(summary: predictive_intelligence.PredictiveSummary, name_for_user: _predictive_intelligence__Callable[[str], str]) -> tuple[str, ...]:
    """Render factual workload outlook lines from currently known tasks only."""
    lines = []
    for row in summary.workload:
        name = name_for_user(row.owner_id) if row.owner_id else 'Unassigned'
        lines.append(f'{name} · {row.pending} pending · {row.due_within_48h} due <48h · {row.due_within_7d} due in 7 days')
    return tuple(lines)
predictive_intelligence.workload_forecast = _predictive_intelligence__workload_forecast


# operations_intelligence.py
'Read-only, explainable operations intelligence over authorized task data.'
from collections import Counter as _operations_intelligence__Counter, defaultdict as _operations_intelligence__defaultdict
operations_intelligence.Counter = _operations_intelligence__Counter
operations_intelligence.defaultdict = _operations_intelligence__defaultdict
from dataclasses import dataclass as _operations_intelligence__dataclass
operations_intelligence.dataclass = _operations_intelligence__dataclass
from datetime import date as _operations_intelligence__date, datetime as _operations_intelligence__datetime, timedelta as _operations_intelligence__timedelta
operations_intelligence.date = _operations_intelligence__date
operations_intelligence.datetime = _operations_intelligence__datetime
operations_intelligence.timedelta = _operations_intelligence__timedelta
import re as _operations_intelligence__re
operations_intelligence.re = _operations_intelligence__re
operations_intelligence.predictive_intelligence = predictive_intelligence
_operations_intelligence__MODES = {'workload', 'risk', 'health', 'heatmap', 'bottlenecks', 'briefing', 'capacity', 'executive', 'meeting', 'unassigned', 'collisions'}
operations_intelligence.MODES = _operations_intelligence__MODES
@_operations_intelligence__dataclass(frozen=True)
class _operations_intelligence__TaskRisk:
    task: object
    level: str
    points: int
    reasons: tuple[str, ...]
operations_intelligence.TaskRisk = _operations_intelligence__TaskRisk
@_operations_intelligence__dataclass(frozen=True)
class _operations_intelligence__Bottleneck:
    title: str
    detail: str
    severity: str
operations_intelligence.Bottleneck = _operations_intelligence__Bottleneck
def _operations_intelligence__parse_request(text):
    """Recognize explicit analytical questions without invoking an LLM."""
    value = _operations_intelligence__re.sub('\\s+', ' ', str(text or '')).strip().rstrip('.?!').casefold()
    if _operations_intelligence__re.fullmatch(
            r"(?:i need you to )?(?:figure out |identify |show )?"
            r"which tasks? (?:are )?(?:putting (?:our |the )?deployment )?at risk",
            value):
        return {'intent': 'operations_intelligence', 'operations_mode': 'risk'}
    if _operations_intelligence__re.fullmatch(
            r"(?:when can we meet|find a meeting time for the team)", value):
        return {'intent': 'operations_intelligence', 'operations_mode': 'meeting'}
    workstream = _operations_intelligence__re.fullmatch('show (testing|deployment|documentation|client|reporting|regression) work', value)
    if workstream:
        theme = workstream.group(1)
        return {'intent': 'list', 'query': theme, 'query_terms': [theme], 'search': True, 'completed': False, 'result_operation': 'return_collection'}
    patterns = (("(?:give me )?(?:my |today'?s? )?daily briefing", 'briefing'), ('(?:give me )?(?:an? )?executive summary', 'executive'), ('(?:give me )?(?:a )?team status report', 'executive'), ('who (?:is|looks) overloaded', 'workload'), ('who has the most overdue work', 'workload'), ('who has the most p1 (?:work|tasks)', 'workload'), ('(?:show )?(?:team )?(?:capacity|workload intelligence)', 'capacity'), ('what are (?:our|the) biggest risks', 'risk'), ('what deadlines are dangerous', 'risk'), ('what is coming up', 'risk'), ('(?:show )?deadline risk', 'risk'), ('(?:show )?task health(?: scores?)?', 'health'), ('(?:show )?(?:the )?deadline heatmap', 'heatmap'), ('where are (?:the )?bottlenecks', 'bottlenecks'), ('(?:show )?(?:operational )?bottlenecks', 'bottlenecks'), ('which tasks are unassigned', 'unassigned'), ('which dates have deadline collisions', 'collisions'), ('when can i meet with .+', 'meeting'), ('find a time for the team', 'meeting'), ('when is everyone available', 'meeting'))
    for pattern, mode in patterns:
        if _operations_intelligence__re.fullmatch(pattern, value):
            result = {'intent': 'operations_intelligence', 'operations_mode': mode}
            person = _operations_intelligence__re.fullmatch('when can i meet with (.+)', value)
            if person:
                result['calendar_member'] = person.group(1).strip()
            return result
    return None
operations_intelligence.parse_request = _operations_intelligence__parse_request
def _operations_intelligence__workload_labels(rows):
    """Label relative visible workload without claiming real employee capacity."""
    active = [row for row in rows if row.pending]
    if not active:
        return {}
    scores = {row.owner_id: row.pending + 2 * row.overdue + 2 * row.p1 for row in active}
    ordered = sorted(scores.values())
    median = ordered[len(ordered) // 2]
    labels = {}
    for owner_id, score in scores.items():
        if score >= median + 3 and score >= 6:
            labels[owner_id] = 'High'
        elif score <= max(1, median - 3):
            labels[owner_id] = 'Low'
        else:
            labels[owner_id] = 'Medium'
    return labels
operations_intelligence.workload_labels = _operations_intelligence__workload_labels
def _operations_intelligence__assess_risks(tasks, today):
    """Return transparent deterministic task risk assessments."""
    values = list(tasks)
    workloads = {row.owner_id: row for row in predictive_intelligence.calculate_workload_pressure(values, today)}
    clusters = predictive_intelligence.calculate_deadline_concentration(values, today, minimum_tasks=2, horizon_days=30)
    cluster_by_task = {task_id: cluster for cluster in clusters for task_id in cluster.task_ids}
    risks = []
    for task in values:
        if task.completed:
            continue
        points, reasons = (0, [])
        if task.due_date and task.due_date < today:
            days = (today - task.due_date).days
            points += 5
            reasons.append(f'Overdue by {days} day{('s' if days != 1 else '')}')
        elif task.due_date:
            days = (task.due_date - today).days
            if days <= 1:
                points += 3
                reasons.append('Due within 24 hours')
            elif days <= 7:
                points += 1
                reasons.append(f'Due in {days} days')
        if task.priority == 'P1':
            points += 3
            reasons.append('P1 priority')
        elif task.priority == 'P2':
            points += 1
            reasons.append('P2 priority')
        if not task.owner_ids:
            points += 2
            reasons.append('No assigned owner')
        owner = workloads.get(task.owner_ids[0]) if len(task.owner_ids) == 1 else None
        if owner and owner.p1 >= 3:
            points += 1
            reasons.append(f'Owner has {owner.p1} pending P1 tasks')
        cluster = cluster_by_task.get(task.item_id)
        if cluster:
            points += 2
            reasons.append(f'{len(cluster.task_ids)} tasks share this deadline')
        level = 'Critical' if points >= 8 else 'High' if points >= 5 else 'Medium' if points >= 2 else 'Low'
        risks.append(operations_intelligence.TaskRisk(task, level, points, tuple(reasons or ('No immediate pressure signal',))))
    rank = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    return tuple(sorted(risks, key=lambda row: (rank[row.level], row.task.due_date or _operations_intelligence__date.max, row.task.name.casefold())))
operations_intelligence.assess_risks = _operations_intelligence__assess_risks
def _operations_intelligence__task_health(risk):
    return 'Critical' if risk.level == 'Critical' else 'At Risk' if risk.level in {'High', 'Medium'} else 'Healthy'
operations_intelligence.task_health = _operations_intelligence__task_health
def _operations_intelligence__heatmap(tasks, today, horizon_days=31):
    grouped = _operations_intelligence__defaultdict(list)
    end = today + _operations_intelligence__timedelta(days=horizon_days)
    for task in tasks:
        if not task.completed and task.due_date and (today <= task.due_date <= end):
            grouped[task.due_date].append(task)
    return tuple(((day, len(values), sum((task.priority == 'P1' for task in values))) for day, values in sorted(grouped.items())))
operations_intelligence.heatmap = _operations_intelligence__heatmap
def _operations_intelligence__bottlenecks(tasks, today, name_for_user):
    values = list(tasks)
    summary = predictive_intelligence.build_predictive_summary(values, today)
    labels = operations_intelligence.workload_labels(summary.workload)
    results = []
    for row in summary.workload:
        if row.owner_id and labels.get(row.owner_id) == 'High':
            results.append(operations_intelligence.Bottleneck(name_for_user(row.owner_id), f'{row.pending} pending, {row.overdue} overdue, {row.p1} P1 tasks.', 'High'))
    unassigned_p1 = sum((not task.owner_ids and task.priority == 'P1' and (not task.completed) for task in values))
    if unassigned_p1:
        results.append(operations_intelligence.Bottleneck('Unassigned high-priority work', f'{unassigned_p1} pending P1 task{('s have' if unassigned_p1 != 1 else ' has')} no owner.', 'High'))
    for cluster in predictive_intelligence.calculate_deadline_concentration(values, today, minimum_tasks=3, horizon_days=30):
        p1 = cluster.priority_counts.get('P1', 0)
        results.append(operations_intelligence.Bottleneck(f'Deadline cluster — {cluster.due_date.strftime('%d %b %Y')}', f'{len(cluster.task_ids)} tasks share this date, including {p1} P1.', 'High' if p1 else 'Medium'))
    return tuple(results)
operations_intelligence.bottlenecks = _operations_intelligence__bottlenecks
def _operations_intelligence__meeting_windows(clocks, requester_clock=None, duration_minutes=60):
    """Find overlap in configured working hours, expressed in requester time."""
    usable = [clock for clock in clocks if clock.local_time and clock.working_start and clock.working_end]
    if len(usable) != len(clocks) or not usable:
        return ()
    requester = requester_clock or usable[0]
    day = requester.local_time.date()
    starts, ends = ([], [])
    for clock in usable:
        local_start = _operations_intelligence__datetime.combine(clock.local_time.date(), clock.working_start, tzinfo=clock.local_time.tzinfo)
        local_end = _operations_intelligence__datetime.combine(clock.local_time.date(), clock.working_end, tzinfo=clock.local_time.tzinfo)
        starts.append(local_start.astimezone(requester.local_time.tzinfo))
        ends.append(local_end.astimezone(requester.local_time.tzinfo))
    start, end = (max(starts), min(ends))
    if start.date() != day or end <= start or end - start < _operations_intelligence__timedelta(minutes=duration_minutes):
        return ()
    return ((start, min(end, start + _operations_intelligence__timedelta(minutes=duration_minutes))),)
operations_intelligence.meeting_windows = _operations_intelligence__meeting_windows


# control_tower.py
'Read-only Action Item Control Tower composed from existing intelligence.'
from dataclasses import dataclass as _control_tower__dataclass
control_tower.dataclass = _control_tower__dataclass
from datetime import date as _control_tower__date
control_tower.date = _control_tower__date
import re as _control_tower__re
control_tower.re = _control_tower__re
control_tower.operations_intelligence = operations_intelligence
control_tower.predictive_intelligence = predictive_intelligence
@_control_tower__dataclass(frozen=True)
class _control_tower__ControlTower:
    total: int
    summary: object
    risks: tuple
    workload_labels: dict
    bottlenecks: tuple
    recommendations: tuple[str, ...]
    unavailable: tuple[str, ...]

    @property
    def overall_workload_pressure(self):
        if 'workload pressure' in self.unavailable:
            return 'Unavailable'
        values = set(self.workload_labels.values())
        return 'High' if 'High' in values else 'Medium' if 'Medium' in values else 'Low'
control_tower.ControlTower = _control_tower__ControlTower
def _control_tower__parse_request(text):
    value = _control_tower__re.sub('\\s+', ' ', str(text or '')).strip().rstrip('.?!').casefold()
    patterns = ('(?:show(?: me)? (?:the )?)?control tower', 'show operations dashboard', 'show operational overview', 'give me the operations overview', 'how are operations doing', 'what is the current operational status')
    if any((_control_tower__re.fullmatch(pattern, value) for pattern in patterns)):
        return {'intent': 'control_tower'}
    return None
control_tower.parse_request = _control_tower__parse_request
def _control_tower___safe_component(name, factory, fallback, unavailable):
    try:
        return factory()
    except Exception:
        unavailable.append(name)
        return fallback
control_tower._safe_component = _control_tower___safe_component
def _control_tower___recommendations(summary, risks, bottlenecks):
    recommendations = []
    for risk in risks:
        if risk.level in {'Critical', 'High'}:
            recommendations.append(f'Review {risk.level.lower()} risk: {risk.task.name}')
        if len(recommendations) >= 2:
            break
    for item in bottlenecks:
        if item.title.startswith('Deadline cluster'):
            recommendations.append(f'Review {item.title.lower()}')
            break
    if summary.unassigned:
        recommendations.append(f'Review {summary.unassigned} unassigned pending task{('s' if summary.unassigned != 1 else '')}')
    active = [row for row in summary.workload if row.owner_id]
    labels = operations_intelligence.workload_labels(active)
    high = next((row for row in active if labels.get(row.owner_id) == 'High'), None)
    if high:
        recommendations.append('Review the highest visible workload concentration')
    if not recommendations and summary.pending:
        recommendations.append('Review the next pending deadline')
    return tuple(dict.fromkeys(recommendations))[:4]
control_tower._recommendations = _control_tower___recommendations
def _control_tower__aggregate(tasks, today: _control_tower__date, name_for_user, *, components=None):
    """Aggregate one authorized task snapshot with section-level isolation."""
    values = tuple(tasks)
    unavailable = []
    components = components or {}
    summary_factory = components.get('summary', lambda: predictive_intelligence.build_predictive_summary(values, today))
    summary = control_tower._safe_component('team health', summary_factory, None, unavailable)
    if summary is None:
        summary = predictive_intelligence.build_predictive_summary((), today)
    risks = control_tower._safe_component('risk radar', components.get('risks', lambda: operations_intelligence.assess_risks(values, today)), (), unavailable)
    labels = control_tower._safe_component('workload pressure', components.get('workload', lambda: operations_intelligence.workload_labels(summary.workload)), {}, unavailable)
    bottlenecks = control_tower._safe_component('bottlenecks', components.get('bottlenecks', lambda: operations_intelligence.bottlenecks(values, today, name_for_user)), (), unavailable)
    recommendations = control_tower._safe_component('recommended actions', components.get('recommendations', lambda: control_tower._recommendations(summary, risks, bottlenecks)), (), unavailable)
    return control_tower.ControlTower(total=len(values), summary=summary, risks=tuple(risks), workload_labels=dict(labels), bottlenecks=tuple(bottlenecks), recommendations=tuple(recommendations), unavailable=tuple(unavailable))
control_tower.aggregate = _control_tower__aggregate


# team_calendar.py
'Read-only team calendar, deadline intelligence, and global team clock.'
from dataclasses import dataclass as _team_calendar__dataclass
team_calendar.dataclass = _team_calendar__dataclass
from datetime import date as _team_calendar__date, datetime as _team_calendar__datetime, time as _team_calendar__time, timedelta as _team_calendar__timedelta, timezone as _team_calendar__timezone
team_calendar.date = _team_calendar__date
team_calendar.datetime = _team_calendar__datetime
team_calendar.time = _team_calendar__time
team_calendar.timedelta = _team_calendar__timedelta
team_calendar.timezone = _team_calendar__timezone
import json as _team_calendar__json
team_calendar.json = _team_calendar__json
import os as _team_calendar__os
team_calendar.os = _team_calendar__os
import re as _team_calendar__re
team_calendar.re = _team_calendar__re
from zoneinfo import ZoneInfo as _team_calendar__ZoneInfo, ZoneInfoNotFoundError as _team_calendar__ZoneInfoNotFoundError
team_calendar.ZoneInfo = _team_calendar__ZoneInfo
team_calendar.ZoneInfoNotFoundError = _team_calendar__ZoneInfoNotFoundError
@_team_calendar__dataclass(frozen=True)
class _team_calendar__MemberClock:
    user_id: str
    name: str
    timezone_name: str | None
    location: str | None
    local_time: _team_calendar__datetime | None
    utc_offset: str | None
    working_start: _team_calendar__time | None
    working_end: _team_calendar__time | None
    availability: str
team_calendar.MemberClock = _team_calendar__MemberClock
def _team_calendar__parse_request(text):
    """Recognize explicit calendar/clock language deterministically."""
    value = _team_calendar__re.sub('\\s+', ' ', str(text or '')).strip().rstrip('.?!')
    lower = value.casefold()
    if _team_calendar__re.fullmatch(r"what is on the calendar today", lower):
        return {'intent': 'calendar', 'calendar_mode': 'today'}
    if _team_calendar__re.fullmatch(
            r"what deadlines are coming this week", lower):
        return {'intent': 'calendar', 'calendar_mode': 'week'}
    exact_modes = (('(?:show\\s+)?(?:the\\s+)?team\\s+calendar', 'team'), ('(?:show\\s+)?(?:my\\s+|the\\s+)?calendar', 'week'), ("(?:show\\s+)?today(?:'s)?\\s+(?:calendar|deadlines)", 'today'), ('(?:show\\s+)?(?:the\\s+)?(?:weekly|week)\\s+calendar', 'week'), ('(?:show\\s+)?(?:the\\s+)?(?:monthly|month)\\s+calendar', 'month'), ('(?:show\\s+)?(?:the\\s+)?calendar\\s+for\\s+upcoming\\s+deadlines', 'upcoming'), ('(?:show\\s+)?overdue\\s+calendar', 'overdue'), ('what\\s+does\\s+the\\s+upcoming\\s+week\\s+look\\s+like', 'week'), ('who\\s+has\\s+deadlines\\s+tomorrow', 'tomorrow'), ('(?:show\\s+)?calendar\\s+deadline\\s+(?:conflicts|pressure)', 'pressure'), ('(?:show\\s+)?(?:the\\s+)?(?:team\\s+)?time\\s*zones?', 'clock'), ('(?:show\\s+)?(?:the\\s+)?(?:global\\s+)?team\\s+clock', 'clock'), ('who\\s+is\\s+(?:currently\\s+)?(?:working|within\\s+working\\s+hours)(?:\\s+right\\s+now)?', 'availability'), ('who\\s+is\\s+outside\\s+working\\s+hours', 'availability'), ('when\\s+can\\s+i\\s+meet\\s+with\\s+(?:the\\s+)?team', 'coordination'), ('what\\s+is\\s+the\\s+best\\s+time\\s+to\\s+coordinate\\s+with\\s+(?:the\\s+)?team', 'coordination'))
    for pattern, mode in exact_modes:
        if _team_calendar__re.fullmatch(pattern, lower):
            return {'intent': 'calendar', 'calendar_mode': mode}
    person_time = _team_calendar__re.fullmatch('what\\s+time\\s+is\\s+it\\s+for\\s+(.+)', value, _team_calendar__re.I)
    if person_time:
        return {'intent': 'calendar', 'calendar_mode': 'clock', 'calendar_member': person_time.group(1).strip()}
    return None
team_calendar.parse_request = _team_calendar__parse_request
def _team_calendar___json_env(name):
    try:
        value = _team_calendar__json.loads(_team_calendar__os.getenv(name, '{}') or '{}')
    except _team_calendar__json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
team_calendar._json_env = _team_calendar___json_env
def _team_calendar___configured(mapping, user_id, name):
    for key in (user_id, name, str(name).casefold()):
        if key in mapping:
            return mapping[key]
    folded = {str(key).casefold(): value for key, value in mapping.items()}
    return folded.get(str(name).casefold())
team_calendar._configured = _team_calendar___configured
def _team_calendar___clock_offset(value):
    offset = value.utcoffset()
    if offset is None:
        return None
    minutes = int(offset.total_seconds() // 60)
    sign = '+' if minutes >= 0 else '−'
    minutes = abs(minutes)
    return f'UTC{sign}{minutes // 60}:{minutes % 60:02d}'
team_calendar._clock_offset = _team_calendar___clock_offset
def _team_calendar___parse_working_hours(value):
    if isinstance(value, str) and '-' in value:
        start, end = value.split('-', 1)
    elif isinstance(value, dict):
        start, end = (value.get('start'), value.get('end'))
    else:
        return (None, None)
    try:
        return (_team_calendar__time.fromisoformat(str(start).strip()), _team_calendar__time.fromisoformat(str(end).strip()))
    except (TypeError, ValueError):
        return (None, None)
team_calendar._parse_working_hours = _team_calendar___parse_working_hours
def _team_calendar___availability(local, start, end):
    if not local or not start or (not end):
        return 'Working hours not configured'
    current = local.time().replace(second=0, microsecond=0)
    inside = start <= current < end if start <= end else current >= start or current < end
    current_minutes = current.hour * 60 + current.minute
    start_minutes = start.hour * 60 + start.minute
    end_minutes = end.hour * 60 + end.minute
    near = min(abs(current_minutes - start_minutes), abs(current_minutes - end_minutes)) <= 60
    if near:
        return 'Near working-hours boundary'
    return 'Working hours' if inside else 'Outside working hours'
team_calendar._availability = _team_calendar___availability
def _team_calendar__member_clocks(members, now=None):
    now = now or _team_calendar__datetime.now(_team_calendar__timezone.utc)
    zones = team_calendar._json_env('TEAM_TIMEZONES_JSON')
    locations = team_calendar._json_env('TEAM_LOCATIONS_JSON')
    hours = team_calendar._json_env('TEAM_WORKING_HOURS_JSON')
    clocks = []
    for member in members:
        profile = member.get('profile') or {}
        user_id = str(member.get('id') or '')
        name = profile.get('display_name') or member.get('real_name') or profile.get('real_name') or member.get('name') or 'Team member'
        zone_name = team_calendar._configured(zones, user_id, name) or member.get('tz') or profile.get('tz')
        local = None
        if zone_name:
            try:
                local = now.astimezone(_team_calendar__ZoneInfo(str(zone_name)))
            except _team_calendar__ZoneInfoNotFoundError:
                zone_name = None
        start, end = team_calendar._parse_working_hours(team_calendar._configured(hours, user_id, name))
        clocks.append(team_calendar.MemberClock(user_id, name, str(zone_name) if zone_name else None, team_calendar._configured(locations, user_id, name) or member.get('tz_label'), local, team_calendar._clock_offset(local) if local else None, start, end, team_calendar._availability(local, start, end)))
    return clocks
team_calendar.member_clocks = _team_calendar__member_clocks
def _team_calendar__filter_tasks(tasks, mode, today):
    pending = [task for task in tasks if not task.completed]
    if mode == 'today':
        return [task for task in pending if task.due_date == today]
    if mode == 'tomorrow':
        return [task for task in pending if task.due_date == today + _team_calendar__timedelta(days=1)]
    if mode == 'week':
        return [task for task in pending if task.due_date and today <= task.due_date <= today + _team_calendar__timedelta(days=6)]
    if mode == 'month':
        return [task for task in pending if task.due_date and (task.due_date.year, task.due_date.month) == (today.year, today.month)]
    if mode == 'overdue':
        return [task for task in pending if task.due_date and task.due_date < today]
    if mode == 'upcoming':
        return [task for task in pending if task.due_date and today <= task.due_date <= today + _team_calendar__timedelta(days=30)]
    return [task for task in pending if task.due_date]
team_calendar.filter_tasks = _team_calendar__filter_tasks
def _team_calendar__render_calendar(tasks, all_tasks, mode, today, name_for_user, *, show_owner=True, show_priority=True):
    labels = {'today': 'TODAY', 'tomorrow': 'TOMORROW', 'week': 'THIS WEEK', 'month': today.strftime('%B %Y').upper(), 'upcoming': 'UPCOMING 30 DAYS', 'overdue': 'OVERDUE', 'team': 'TEAM', 'pressure': 'DEADLINE PRESSURE'}
    selected = sorted(team_calendar.filter_tasks(tasks, mode, today), key=lambda task: (task.due_date or _team_calendar__date.max, {'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4}.get(task.priority, 9), task.name.casefold()))
    pending = [task for task in all_tasks if not task.completed]
    overdue = [task for task in pending if task.due_date and task.due_date < today]
    due_today = [task for task in pending if task.due_date == today]
    due_week = [task for task in pending if task.due_date and today <= task.due_date <= today + _team_calendar__timedelta(days=6)]
    p1 = [task for task in selected if show_priority and task.priority == 'P1']
    lines = [f'*TEAM CALENDAR — {labels.get(mode, 'TEAM')}*', '', f'*{len(selected)} deadlines* · {len(overdue)} overdue · {len(due_today)} due today · {len(due_week)} this week']
    if not selected:
        lines.extend(('', '_No authorized deadlines match this calendar view._'))
        return '\n'.join(lines)
    grouped = {}
    for task in selected:
        grouped.setdefault(task.due_date, []).append(task)
    for due, values in grouped.items():
        if due < today:
            day_label = f'OVERDUE · {due.strftime('%a %b %d').upper()}'
        elif due == today:
            day_label = f'TODAY · {due.strftime('%a %b %d').upper()}'
        elif due == today + _team_calendar__timedelta(days=1):
            day_label = f'TOMORROW · {due.strftime('%a %b %d').upper()}'
        else:
            day_label = due.strftime('%A · %b %d').upper()
        lines.extend(('', f'*{day_label}*'))
        for task in values[:12]:
            priority_label = f'[{task.priority}]' if show_priority and task.priority else '[TASK]'
            owners = ', '.join((name_for_user(owner) for owner in task.owner_ids)) or 'Unassigned' if show_owner else 'Owner restricted'
            status = 'Overdue' if due < today else 'Due today' if due == today else task.priority if show_priority else 'Upcoming'
            lines.append(f'{priority_label} *{task.name}*\n   {owners} · {status}')
        if len(values) > 12:
            lines.append(f'_…and {len(values) - 12} more deadlines_ ')
    collisions = [(due, values) for due, values in grouped.items() if len(values) >= 3]
    pressure = {}
    if show_owner:
        for task in selected:
            for owner in task.owner_ids or ('unassigned',):
                weight = 3 if show_priority and task.priority == 'P1' else 2 if show_priority and task.priority == 'P2' else 1
                pressure[owner] = pressure.get(owner, 0) + weight
    lines.extend(('', '*DEADLINE INTELLIGENCE*', f'• {len(p1)} high-priority deadline{('s' if len(p1) != 1 else '')} in this view' if show_priority else '• Priority details are restricted for this view', f'• {len(collisions)} date collision{('s' if len(collisions) != 1 else '')} with 3+ tasks'))
    if pressure:
        owner, score = max(pressure.items(), key=lambda row: row[1])
        label = 'Unassigned' if owner == 'unassigned' else name_for_user(owner)
        lines.append(f'• Highest deadline pressure: *{label}* · score {score}')
    lines.append('\n_Read-only calendar · No task changes were made._')
    return '\n'.join(lines)
team_calendar.render_calendar = _team_calendar__render_calendar
def _team_calendar__render_clock(clocks, requester_clock=None, coordination=False):
    title = '*TEAM TIME ZONES*'
    lines = [title, '', f'*{len(clocks)} team members* · Time-zone data is never guessed']
    configured = []
    for clock in clocks:
        lines.extend(('', f'*{clock.name}*'))
        if not clock.timezone_name or not clock.local_time:
            lines.append('   Time zone not configured')
            continue
        configured.append(clock)
        relative = ''
        if requester_clock and requester_clock.local_time and (requester_clock.user_id != clock.user_id):
            delta_minutes = int((clock.local_time.utcoffset() - requester_clock.local_time.utcoffset()).total_seconds() // 60)
            if delta_minutes:
                sign = '+' if delta_minutes > 0 else '−'
                absolute = abs(delta_minutes)
                relative = f' · {sign}{absolute // 60}h {absolute % 60:02d}m vs you'
            else:
                relative = ' · Same offset as you'
        lines.append(f'   {clock.local_time.strftime('%a · %b %d · %I:%M %p')} · {clock.utc_offset}{relative}')
        location = f' · {clock.location}' if clock.location else ''
        lines.append(f'   {clock.timezone_name}{location}')
        if clock.working_start and clock.working_end:
            lines.append(f'   {clock.working_start.strftime('%I:%M %p')}–{clock.working_end.strftime('%I:%M %p')} · {clock.availability}')
        else:
            lines.append('   Working hours not configured')
    if coordination:
        lines.extend(('', '*⏰ Coordination Window*'))
        if configured and all((clock.working_start and clock.working_end for clock in configured)):
            now = _team_calendar__datetime.now(_team_calendar__timezone.utc).replace(second=0, microsecond=0)
            found = None
            for step in range(0, 7 * 48):
                candidate = now + _team_calendar__timedelta(minutes=30 * step)
                if all((clock.working_start <= candidate.astimezone(_team_calendar__ZoneInfo(clock.timezone_name)).time() < clock.working_end for clock in configured)):
                    found = candidate
                    break
            lines.append(found.strftime('• Earliest shared working window starts %a %b %d at %H:%M UTC') if found else '• No shared configured working window was found in the next 7 days.')
        else:
            lines.append("• Configure every member's time zone and working hours to calculate an overlap.")
    lines.append('\n_Read-only team clock · No locations or time zones were inferred._')
    return '\n'.join(lines)
team_calendar.render_clock = _team_calendar__render_clock


# action_item_sentinel.py
'Deterministic, persistent monitoring over normalized Slack List tasks.'
import hashlib as _action_item_sentinel__hashlib
action_item_sentinel.hashlib = _action_item_sentinel__hashlib
import json as _action_item_sentinel__json
action_item_sentinel.json = _action_item_sentinel__json
import logging as _action_item_sentinel__logging
action_item_sentinel.logging = _action_item_sentinel__logging
import os as _action_item_sentinel__os
action_item_sentinel.os = _action_item_sentinel__os
import sqlite3 as _action_item_sentinel__sqlite3
action_item_sentinel.sqlite3 = _action_item_sentinel__sqlite3
import threading as _action_item_sentinel__threading
action_item_sentinel.threading = _action_item_sentinel__threading
import time as _action_item_sentinel__time
action_item_sentinel.time = _action_item_sentinel__time
from dataclasses import asdict as _action_item_sentinel__asdict, dataclass as _action_item_sentinel__dataclass
action_item_sentinel.asdict = _action_item_sentinel__asdict
action_item_sentinel.dataclass = _action_item_sentinel__dataclass
from datetime import date as _action_item_sentinel__date, datetime as _action_item_sentinel__datetime, timedelta as _action_item_sentinel__timedelta
action_item_sentinel.date = _action_item_sentinel__date
action_item_sentinel.datetime = _action_item_sentinel__datetime
action_item_sentinel.timedelta = _action_item_sentinel__timedelta
from pathlib import Path as _action_item_sentinel__Path
action_item_sentinel.Path = _action_item_sentinel__Path
from typing import Iterable as _action_item_sentinel__Iterable
action_item_sentinel.Iterable = _action_item_sentinel__Iterable
action_item_sentinel.project_intelligence = project_intelligence
action_item_sentinel.predictive_intelligence = predictive_intelligence
_action_item_sentinel__logger = _action_item_sentinel__logging.getLogger('slack_list.sentinel')
action_item_sentinel.logger = _action_item_sentinel__logger
def _action_item_sentinel___env_bool(name: str, default: bool) -> bool:
    value = _action_item_sentinel__os.getenv(name)
    return default if value is None else value.strip().casefold() in {'1', 'true', 'yes', 'on'}
action_item_sentinel._env_bool = _action_item_sentinel___env_bool
def _action_item_sentinel___env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(_action_item_sentinel__os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))
action_item_sentinel._env_int = _action_item_sentinel___env_int
@_action_item_sentinel__dataclass(frozen=True)
class _action_item_sentinel__SentinelSettings:
    enabled: bool = True
    warning_days: int = 2
    combined_task_threshold: int = 2
    max_alerts_per_scan: int = 100
    approval_ttl_minutes: int = 30

    @classmethod
    def from_env(cls):
        return cls(enabled=action_item_sentinel._env_bool('ACTION_ITEM_SENTINEL_ENABLED', True), warning_days=action_item_sentinel._env_int('SENTINEL_WARNING_DAYS', 2, 1, 14), combined_task_threshold=action_item_sentinel._env_int('SENTINEL_COMBINED_TASK_THRESHOLD', 2, 2, 20), max_alerts_per_scan=action_item_sentinel._env_int('SENTINEL_MAX_ALERTS_PER_SCAN', 100, 1, 1000), approval_ttl_minutes=action_item_sentinel._env_int('SENTINEL_APPROVAL_TTL_MINUTES', 30, 1, 1440))
action_item_sentinel.SentinelSettings = _action_item_sentinel__SentinelSettings
@_action_item_sentinel__dataclass(frozen=True)
class _action_item_sentinel__TaskChange:
    task_id: str
    task_name: str
    change_type: str
    previous_value: object
    current_value: object
    timestamp: float
action_item_sentinel.TaskChange = _action_item_sentinel__TaskChange
@_action_item_sentinel__dataclass(frozen=True)
class _action_item_sentinel__SentinelRisk:
    task_id: str
    task_ids: tuple[str, ...]
    task_name: str
    risk_type: str
    severity: str
    reasons: tuple[str, ...]
    owner_ids: tuple[str, ...]
    priority: str | None
    due_date: _action_item_sentinel__date | None
    recommendation: str
action_item_sentinel.SentinelRisk = _action_item_sentinel__SentinelRisk
@_action_item_sentinel__dataclass(frozen=True)
class _action_item_sentinel__SentinelAlert:
    alert_id: str
    list_id: str
    task_id: str
    event_type: str
    state_hash: str
    severity: str
    payload: dict
    status: str
    created_at: float
action_item_sentinel.SentinelAlert = _action_item_sentinel__SentinelAlert
@_action_item_sentinel__dataclass(frozen=True)
class _action_item_sentinel__SentinelEvaluation:
    changes: tuple[action_item_sentinel.TaskChange, ...]
    risks: tuple[action_item_sentinel.SentinelRisk, ...]
    emitted: tuple[action_item_sentinel.SentinelAlert, ...]
    suppressed: int
action_item_sentinel.SentinelEvaluation = _action_item_sentinel__SentinelEvaluation
def _action_item_sentinel__task_state(task: project_intelligence.NormalizedTask) -> dict:
    return {'task_id': task.item_id, 'name': task.name, 'owner_ids': list(task.owner_ids), 'priority': task.priority, 'due_date': task.due_date.isoformat() if task.due_date else None, 'completed': task.completed, 'status': task.status}
action_item_sentinel.task_state = _action_item_sentinel__task_state
def _action_item_sentinel__task_state_hash(task: project_intelligence.NormalizedTask) -> str:
    return _action_item_sentinel__hashlib.sha256(_action_item_sentinel__json.dumps(action_item_sentinel.task_state(task), sort_keys=True).encode()).hexdigest()
action_item_sentinel.task_state_hash = _action_item_sentinel__task_state_hash
def _action_item_sentinel__detect_changes(previous: dict[str, dict], current: _action_item_sentinel__Iterable[project_intelligence.NormalizedTask], timestamp: float | None=None) -> list[action_item_sentinel.TaskChange]:
    """Compare reliable snapshots and return only meaningful transitions."""
    timestamp = _action_item_sentinel__time.time() if timestamp is None else timestamp
    current_by_id = {task.item_id: task for task in current}
    changes = []
    for task_id, task in current_by_id.items():
        before = previous.get(task_id)
        after = action_item_sentinel.task_state(task)
        if before is None:
            changes.append(action_item_sentinel.TaskChange(task_id, task.name, 'created', None, after, timestamp))
            continue
        comparisons = (('completed' if after['completed'] else 'reopened', 'completed'), ('priority_changed', 'priority'), ('due_date_changed', 'due_date'), ('owner_changed', 'owner_ids'))
        for change_type, field in comparisons:
            if before.get(field) == after.get(field):
                continue
            if field == 'priority' and after.get(field) == 'P1':
                change_type = 'priority_escalated'
            elif field == 'due_date':
                earlier, later = (before.get(field), after.get(field))
                change_type = 'deadline_moved_earlier' if earlier and later and (later < earlier) else 'deadline_moved_later'
            elif field == 'owner_ids':
                if not before.get(field) and after.get(field):
                    change_type = 'assigned'
                elif before.get(field) and (not after.get(field)):
                    change_type = 'unassigned'
            changes.append(action_item_sentinel.TaskChange(task_id, task.name, change_type, before.get(field), after.get(field), timestamp))
    for task_id, before in previous.items():
        if task_id not in current_by_id:
            changes.append(action_item_sentinel.TaskChange(task_id, before.get('name') or 'Action item', 'deleted', before, None, timestamp))
    return changes
action_item_sentinel.detect_changes = _action_item_sentinel__detect_changes
def _action_item_sentinel__detect_risks(snapshot: _action_item_sentinel__Iterable[project_intelligence.NormalizedTask], today: _action_item_sentinel__date, warning_days: int=2, combined_task_threshold: int=2) -> list[action_item_sentinel.SentinelRisk]:
    """Detect explainable deadline and workload risks from one snapshot."""
    tasks = [task for task in snapshot if not task.completed]
    base_ids = {record.item_id for record in project_intelligence.analyze_task_risks(tasks, {}, today)}
    risks = []
    for task in tasks:
        due_delta = (task.due_date - today).days if task.due_date else None
        reasons = []
        risk_type = None
        severity = 'attention'
        recommendation = 'Confirm completion status or update the deadline.'
        if due_delta is not None and due_delta < 0:
            risk_type, severity = ('overdue', 'critical')
            reasons.extend((f'Overdue by {-due_delta} day{('s' if due_delta != -1 else '')}', 'Still pending'))
            if task.priority == 'P1':
                reasons.insert(0, 'P1 priority')
        elif not task.owner_ids and task.priority == 'P1' and (due_delta is not None) and (due_delta <= warning_days):
            risk_type = 'unassigned_deadline_risk'
            reasons.extend(('P1 priority', f'Deadline within {warning_days * 24} hours', 'No assigned owner', 'Still pending'))
            recommendation = 'Assign an owner or update the deadline.'
        elif task.priority == 'P1' and due_delta is not None and (due_delta <= 1):
            risk_type = 'deadline_risk'
            reasons.extend(('P1 priority', 'Deadline within 24 hours', 'Still pending'))
        if risk_type and (task.item_id in base_ids or risk_type == 'unassigned_deadline_risk'):
            risks.append(action_item_sentinel.SentinelRisk(task.item_id, (task.item_id,), task.name, risk_type, severity, tuple(reasons), task.owner_ids, task.priority, task.due_date, recommendation))
    urgent_by_owner = {}
    for task in tasks:
        delta = (task.due_date - today).days if task.due_date else None
        if task.priority == 'P1' and delta is not None and (delta <= warning_days):
            for owner_id in task.owner_ids:
                urgent_by_owner.setdefault(owner_id, []).append(task)
    for owner_id, urgent in urgent_by_owner.items():
        if len(urgent) < combined_task_threshold:
            continue
        risks.append(action_item_sentinel.SentinelRisk(f'owner:{owner_id}', tuple((task.item_id for task in urgent)), 'Multiple urgent action items', 'combined_workload_risk', 'attention', (f'{len(urgent)} P1 tasks assigned to the same owner', f'Deadlines fall within {warning_days * 24} hours'), (owner_id,), 'P1', min((task.due_date for task in urgent if task.due_date)), 'Review priorities, ownership, and delivery dates.'))
    represented = {task_id for risk in risks if risk.risk_type != 'combined_workload_risk' for task_id in risk.task_ids}
    for emerging in predictive_intelligence.detect_emerging_risks(tasks, today):
        if emerging.task_id in represented:
            continue
        risks.append(action_item_sentinel.SentinelRisk(emerging.task_id, (emerging.task_id,), emerging.task_name, 'emerging_predictive_risk', 'attention', emerging.evidence, emerging.owner_ids, emerging.priority, emerging.due_date, 'Review the task before deadline pressure increases.'))
    return risks
action_item_sentinel.detect_risks = _action_item_sentinel__detect_risks
class _action_item_sentinel__SentinelStore:
    """Persistent snapshots, deduplicated alerts, and approval audit."""

    def __init__(self, path: str):
        self.path = str(_action_item_sentinel__Path(path))
        self._lock = _action_item_sentinel__threading.Lock()
        with self._connect() as connection:
            connection.executescript('\n                CREATE TABLE IF NOT EXISTS sentinel_snapshots (\n                    list_id TEXT NOT NULL, task_id TEXT NOT NULL, state_json TEXT NOT NULL,\n                    seen_at REAL NOT NULL, PRIMARY KEY (list_id, task_id));\n                CREATE TABLE IF NOT EXISTS sentinel_alerts (\n                    alert_id TEXT PRIMARY KEY, list_id TEXT NOT NULL, task_id TEXT NOT NULL,\n                    event_type TEXT NOT NULL, state_hash TEXT NOT NULL, severity TEXT NOT NULL,\n                    payload_json TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL,\n                    acted_at REAL, actor_id TEXT);\n                CREATE TABLE IF NOT EXISTS sentinel_displays (\n                    list_id TEXT NOT NULL, channel_id TEXT NOT NULL, user_id TEXT NOT NULL,\n                    position INTEGER NOT NULL, alert_id TEXT NOT NULL, displayed_at REAL NOT NULL,\n                    PRIMARY KEY (list_id, channel_id, user_id, position));\n            ')

    def _connect(self):
        return _action_item_sentinel__sqlite3.connect(self.path, timeout=10)

    def previous(self, list_id: str) -> dict[str, dict]:
        with self._connect() as connection:
            rows = connection.execute('SELECT task_id, state_json FROM sentinel_snapshots WHERE list_id=?', (list_id,)).fetchall()
        return {task_id: _action_item_sentinel__json.loads(value) for task_id, value in rows}

    def save_snapshot(self, list_id: str, snapshot, seen_at: float):
        current = {task.item_id: action_item_sentinel.task_state(task) for task in snapshot}
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('DELETE FROM sentinel_snapshots WHERE list_id=?', (list_id,))
            connection.executemany('INSERT INTO sentinel_snapshots VALUES (?, ?, ?, ?)', [(list_id, task_id, _action_item_sentinel__json.dumps(state, sort_keys=True), seen_at) for task_id, state in current.items()])

    def add_alert(self, list_id: str, task_id: str, event_type: str, state_hash: str, severity: str, payload: dict, created_at: float) -> tuple[action_item_sentinel.SentinelAlert, bool]:
        alert_id = _action_item_sentinel__hashlib.sha256(f'{list_id}|{task_id}|{event_type}|{state_hash}'.encode()).hexdigest()
        with self._lock, self._connect() as connection:
            inserted = connection.execute("INSERT OR IGNORE INTO sentinel_alerts VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, NULL, NULL)", (alert_id, list_id, task_id, event_type, state_hash, severity, _action_item_sentinel__json.dumps(payload, sort_keys=True), created_at)).rowcount == 1
            row = connection.execute('SELECT status, created_at FROM sentinel_alerts WHERE alert_id=?', (alert_id,)).fetchone()
        return (action_item_sentinel.SentinelAlert(alert_id, list_id, task_id, event_type, state_hash, severity, payload, row[0], row[1]), inserted)

    def resolve_inactive_risks(self, list_id: str, active_ids: set[str], now: float):
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT alert_id FROM sentinel_alerts WHERE list_id=? AND status='active' AND event_type LIKE 'risk:%'", (list_id,)).fetchall()
            for alert_id, in rows:
                if alert_id not in active_ids:
                    connection.execute("UPDATE sentinel_alerts SET status='resolved', acted_at=? WHERE alert_id=?", (now, alert_id))

    def alerts(self, list_id: str, task_ids=(), status='active', since=None, limit=100):
        query = 'SELECT alert_id, task_id, event_type, state_hash, severity, payload_json, status, created_at FROM sentinel_alerts WHERE list_id=?'
        params = [list_id]
        if status:
            query += ' AND status=?'
            params.append(status)
        if task_ids:
            placeholders = ','.join(('?' for _ in task_ids))
            query += f' AND task_id IN ({placeholders})'
            params.extend(task_ids)
        if since is not None:
            query += ' AND created_at>=?'
            params.append(float(since))
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [action_item_sentinel.SentinelAlert(row[0], list_id, row[1], row[2], row[3], row[4], _action_item_sentinel__json.loads(row[5]), row[6], row[7]) for row in rows]

    def alert(self, alert_id: str) -> action_item_sentinel.SentinelAlert | None:
        """Load one alert regardless of status for idempotent action responses."""
        with self._connect() as connection:
            row = connection.execute('SELECT list_id, task_id, event_type, state_hash, severity, payload_json, status, created_at FROM sentinel_alerts WHERE alert_id=?', (alert_id,)).fetchone()
        if not row:
            return None
        return action_item_sentinel.SentinelAlert(alert_id, row[0], row[1], row[2], row[3], row[4], _action_item_sentinel__json.loads(row[5]), row[6], row[7])

    def save_display(self, list_id: str, channel_id: str, user_id: str, alert_ids: _action_item_sentinel__Iterable[str], displayed_at: float):
        """Persist the latest numbered alert view for one authenticated viewer."""
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('DELETE FROM sentinel_displays WHERE list_id=? AND channel_id=? AND user_id=?', (list_id, channel_id, user_id))
            connection.executemany('INSERT INTO sentinel_displays VALUES (?, ?, ?, ?, ?, ?)', [(list_id, channel_id, user_id, position, alert_id, displayed_at) for position, alert_id in enumerate(alert_ids, 1)])

    def resolve_display(self, list_id: str, channel_id: str, user_id: str, position: int) -> tuple[str, float] | None:
        with self._connect() as connection:
            row = connection.execute('SELECT alert_id, displayed_at FROM sentinel_displays WHERE list_id=? AND channel_id=? AND user_id=? AND position=?', (list_id, channel_id, user_id, position)).fetchone()
        return (row[0], row[1]) if row else None

    def claim_action(self, alert_id: str, actor_id: str, expected_hash: str) -> bool:
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT status, state_hash FROM sentinel_alerts WHERE alert_id=?', (alert_id,)).fetchone()
            if not row or row[0] != 'active' or row[1] != expected_hash:
                return False
            connection.execute("UPDATE sentinel_alerts SET status='sending', acted_at=?, actor_id=? WHERE alert_id=?", (_action_item_sentinel__time.time(), actor_id, alert_id))
        return True

    def finish_action(self, alert_id: str, status: str):
        with self._lock, self._connect() as connection:
            connection.execute('UPDATE sentinel_alerts SET status=?, acted_at=? WHERE alert_id=?', (status, _action_item_sentinel__time.time(), alert_id))

    def dismiss(self, alert_id: str, actor_id: str) -> bool:
        with self._lock, self._connect() as connection:
            updated = connection.execute("UPDATE sentinel_alerts SET status='dismissed', acted_at=?, actor_id=? WHERE alert_id=? AND status='active'", (_action_item_sentinel__time.time(), actor_id, alert_id)).rowcount
        return bool(updated)
action_item_sentinel.SentinelStore = _action_item_sentinel__SentinelStore
class _action_item_sentinel__ActionItemSentinel:

    def __init__(self, store: action_item_sentinel.SentinelStore, settings: action_item_sentinel.SentinelSettings | None=None):
        self.store = store
        self.settings = settings or action_item_sentinel.SentinelSettings.from_env()

    def evaluate(self, list_id: str, snapshot, now: _action_item_sentinel__datetime, persist_snapshot: bool=True) -> action_item_sentinel.SentinelEvaluation:
        if not self.settings.enabled:
            return action_item_sentinel.SentinelEvaluation((), (), (), 0)
        snapshot = list(snapshot)
        action_item_sentinel.logger.info('sentinel_evaluation_started list_id=%s task_count=%d llm_used=false', list_id, len(snapshot))
        previous = self.store.previous(list_id) if persist_snapshot else {}
        changes = action_item_sentinel.detect_changes(previous, snapshot, now.timestamp()) if persist_snapshot and previous else []
        risks = action_item_sentinel.detect_risks(snapshot, now.date(), self.settings.warning_days, self.settings.combined_task_threshold)
        emitted, suppressed, active_risk_ids = ([], 0, set())
        events = []
        for change in changes:
            current = next((task for task in snapshot if task.item_id == change.task_id), None)
            state_hash = action_item_sentinel.task_state_hash(current) if current else _action_item_sentinel__hashlib.sha256(_action_item_sentinel__json.dumps(change.previous_value, sort_keys=True).encode()).hexdigest()
            events.append((change.task_id, f'change:{change.change_type}', state_hash, 'attention', _action_item_sentinel__asdict(change)))
        for risk in risks:
            risk_tasks = [task for task in snapshot if task.item_id in risk.task_ids]
            state = {'risk': risk.risk_type, 'tasks': [action_item_sentinel.task_state(task) for task in risk_tasks]}
            state_hash = _action_item_sentinel__hashlib.sha256(_action_item_sentinel__json.dumps(state, sort_keys=True).encode()).hexdigest()
            events.append((risk.task_id, f'risk:{risk.risk_type}', state_hash, risk.severity, {**_action_item_sentinel__asdict(risk), 'task_state_hash': action_item_sentinel.task_state_hash(risk_tasks[0]) if len(risk_tasks) == 1 else None, 'due_date': risk.due_date.isoformat() if risk.due_date else None}))
        for task_id, event_type, state_hash, severity, payload in events[:self.settings.max_alerts_per_scan]:
            alert, inserted = self.store.add_alert(list_id, task_id, event_type, state_hash, severity, payload, now.timestamp())
            if event_type.startswith('risk:'):
                active_risk_ids.add(alert.alert_id)
            if inserted:
                emitted.append(alert)
            else:
                suppressed += 1
                action_item_sentinel.logger.info('sentinel_alert_suppressed alert_id=%s reason=duplicate', alert.alert_id)
        if persist_snapshot:
            self.store.resolve_inactive_risks(list_id, active_risk_ids, now.timestamp())
            self.store.save_snapshot(list_id, snapshot, now.timestamp())
        action_item_sentinel.logger.info('sentinel_changes_detected count=%d sentinel_risks_detected count=%d', len(changes), len(risks))
        action_item_sentinel.logger.info('sentinel_alerts_emitted count=%d suppressed=%d llm_used=false llm_call_count=0', len(emitted), suppressed)
        return action_item_sentinel.SentinelEvaluation(tuple(changes), tuple(risks), tuple(emitted), suppressed)
action_item_sentinel.ActionItemSentinel = _action_item_sentinel__ActionItemSentinel


# smart_task_autopilot.py
'Deterministic prepared actions layered on persisted Sentinel alerts.'
import hashlib as _smart_task_autopilot__hashlib
smart_task_autopilot.hashlib = _smart_task_autopilot__hashlib
from dataclasses import dataclass as _smart_task_autopilot__dataclass
smart_task_autopilot.dataclass = _smart_task_autopilot__dataclass
from datetime import date as _smart_task_autopilot__date
smart_task_autopilot.date = _smart_task_autopilot__date
@_smart_task_autopilot__dataclass(frozen=True)
class _smart_task_autopilot__AutopilotRecommendation:
    recommendation_id: str
    alert_id: str
    task_id: str
    requesting_user: str
    target_user: str | None
    action_type: str
    task_state_version: str
    created_at: float
    status: str
    recommendation: str
    prepared_message: str | None
    executable: bool
smart_task_autopilot.AutopilotRecommendation = _smart_task_autopilot__AutopilotRecommendation
def _smart_task_autopilot___deadline_phrase(due_date: _smart_task_autopilot__date | None, today: _smart_task_autopilot__date) -> str:
    if due_date is None:
        return 'has no due date'
    delta = (due_date - today).days
    if delta == -1:
        return 'was due yesterday'
    if delta < -1:
        return f'was due {-delta} days ago'
    if delta == 0:
        return 'is due today'
    if delta == 1:
        return 'is due tomorrow'
    return f'is due on {due_date.strftime('%b %-d')}'
smart_task_autopilot._deadline_phrase = _smart_task_autopilot___deadline_phrase
def _smart_task_autopilot__prepare_recommendation(*, alert_id: str, payload: dict, requesting_user: str, owner_name: str | None, today: _smart_task_autopilot__date, created_at: float, status: str='prepared') -> smart_task_autopilot.AutopilotRecommendation:
    """Prepare one factual next action from a Sentinel alert payload."""
    task_ids = tuple(payload.get('task_ids') or ())
    task_id = task_ids[0] if len(task_ids) == 1 else str(payload.get('task_id') or '')
    owners = tuple(payload.get('owner_ids') or ())
    target_user = owners[0] if owners else None
    risk_type = str(payload.get('risk_type') or '')
    task_name = str(payload.get('task_name') or 'Action item')
    raw_due = payload.get('due_date')
    try:
        due_date = _smart_task_autopilot__date.fromisoformat(str(raw_due)[:10]) if raw_due else None
    except ValueError:
        due_date = None
    if risk_type == 'combined_workload_risk':
        count = len(task_ids)
        action_type = 'review_workload'
        recommendation = f'{owner_name or 'This owner'} — review the {count} P1 task{('s' if count != 1 else '')} and confirm {('their deadlines' if count != 1 else 'its deadline')}.'
        prepared_message = None
        executable = False
    elif risk_type == 'unassigned_deadline_risk' or not owners:
        action_type = 'assign_owner'
        recommendation = 'Assign an owner or review the deadline.'
        prepared_message = None
        executable = False
    elif risk_type == 'overdue':
        action_type = 'send_reminder'
        recommendation = f'Send a reminder to {owner_name or 'the owner'}.'
        prepared_message = f'Hi {owner_name or 'there'}, {task_name} is still pending and {smart_task_autopilot._deadline_phrase(due_date, today)}. Please confirm the status or update the deadline.'
        executable = True
    elif risk_type == 'deadline_risk':
        action_type = 'send_deadline_reminder'
        recommendation = f'Send a deadline reminder to {owner_name or 'the owner'}.'
        prepared_message = f'Hi {owner_name or 'there'}, {task_name} {smart_task_autopilot._deadline_phrase(due_date, today)} and is still pending. Please confirm the completion status or update the deadline.'
        executable = True
    else:
        action_type = 'review_task'
        recommendation = 'No safe automated recommendation is available for this risk.'
        prepared_message = None
        executable = False
    state_version = str(payload.get('task_state_hash') or '')
    recommendation_id = _smart_task_autopilot__hashlib.sha256(f'{alert_id}|{requesting_user}|{action_type}|{state_version}'.encode()).hexdigest()
    return smart_task_autopilot.AutopilotRecommendation(recommendation_id=recommendation_id, alert_id=alert_id, task_id=task_id, requesting_user=requesting_user, target_user=target_user, action_type=action_type, task_state_version=state_version, created_at=created_at, status=status, recommendation=recommendation, prepared_message=prepared_message, executable=executable)
smart_task_autopilot.prepare_recommendation = _smart_task_autopilot__prepare_recommendation


# command_center.py
'Read-only orchestration for a normalized Slack List task snapshot.'
from dataclasses import dataclass as _command_center__dataclass
command_center.dataclass = _command_center__dataclass
from datetime import date as _command_center__date, timedelta as _command_center__timedelta
command_center.date = _command_center__date
command_center.timedelta = _command_center__timedelta
import logging as _command_center__logging
command_center.logging = _command_center__logging
command_center.action_item_sentinel = action_item_sentinel
command_center.project_intelligence = project_intelligence
command_center.predictive_intelligence = predictive_intelligence
_command_center__logger = _command_center__logging.getLogger('slack_list.command_center')
command_center.logger = _command_center__logger
@_command_center__dataclass(frozen=True)
class _command_center__CommandCenterReport:
    pending: int
    completed: int
    priorities: dict[str, int]
    overdue: int
    due_this_week: int
    critical: tuple[project_intelligence.NormalizedTask, ...]
    risks: tuple[action_item_sentinel.SentinelRisk, ...]
    workload: project_intelligence.WorkloadReport
    reminder_count: int
    workload_review_count: int
    insight: str
    next_step: str
    predictive: predictive_intelligence.PredictiveSummary
command_center.CommandCenterReport = _command_center__CommandCenterReport
def _command_center__build_report(snapshot, *, today: _command_center__date, name_for_user) -> command_center.CommandCenterReport:
    """Calculate all Command Center facts from one normalized snapshot."""
    tasks = list(snapshot)
    pending = [task for task in tasks if not task.completed]
    completed = [task for task in tasks if task.completed]
    priorities = {key: sum((task.priority == key for task in pending)) for key in ('P1', 'P2', 'P3')}
    overdue = [task for task in pending if task.due_date and task.due_date < today]
    due_this_week = [task for task in pending if task.due_date and today <= task.due_date <= today + _command_center__timedelta(days=6)]
    critical = sorted([task for task in pending if task in overdue or task.due_date == today], key=lambda task: (task.due_date or _command_center__date.max, project_intelligence.PRIORITY_RANK.get(task.priority, 5), task.name.casefold()))
    risks = action_item_sentinel.detect_risks(tasks, today)
    predictive = predictive_intelligence.build_predictive_summary(tasks, today)
    try:
        workload = project_intelligence.calculate_workload(tasks, {}, name_for_user, today, {owner for task in tasks for owner in task.owner_ids})
    except Exception as exc:
        command_center.logger.error('command_center_section_degraded section=workload error_type=%s', type(exc).__name__)
        workload = project_intelligence.WorkloadReport()
    reminders = sum((risk.risk_type in {'overdue', 'deadline_risk'} and bool(risk.owner_ids) for risk in risks))
    workload_reviews = sum((risk.risk_type == 'combined_workload_risk' for risk in risks))
    unassigned_p1 = sum((task.priority == 'P1' and (not task.owner_ids) for task in pending))
    if overdue:
        insight = 'Overdue work is currently the main source of operational risk.'
    elif any((risk.risk_type == 'deadline_risk' for risk in risks)):
        insight = 'Approaching P1 deadlines are currently the main source of operational risk.'
    elif workload_reviews:
        insight = 'Concentrated high-priority workload is the main current risk signal.'
    elif unassigned_p1:
        insight = 'Unassigned high-priority work requires attention.'
    else:
        insight = 'No active deterministic risk signal currently dominates the task snapshot.'
    if overdue:
        next_step = 'Review the overdue tasks, starting with P1 items.'
    elif any((risk.risk_type == 'deadline_risk' for risk in risks)):
        next_step = 'Review the P1 tasks due within 48 hours.'
    elif workload_reviews:
        next_step = 'Review concentrated P1 workloads and confirm their deadlines.'
    elif unassigned_p1:
        next_step = 'Assign owners to the unassigned P1 tasks.'
    elif pending:
        next_step = 'Review the next pending deadline.'
    else:
        next_step = 'No pending action is required.'
    return command_center.CommandCenterReport(len(pending), len(completed), priorities, len(overdue), len(due_this_week), tuple(critical[:5]), tuple(risks), workload, reminders, workload_reviews, insight, next_step, predictive)
command_center.build_report = _command_center__build_report
def _command_center__owner_risks(snapshot, owner_id: str, *, today: _command_center__date):
    """Return factual owner risks without inspecting unauthorized tasks."""
    owned = [task for task in snapshot if not task.completed and owner_id in task.owner_ids]
    task_ids = {task.item_id for task in owned}
    risks = action_item_sentinel.detect_risks(snapshot, today)
    return (owned, [risk for risk in risks if task_ids.intersection(risk.task_ids) or owner_id in risk.owner_ids])
command_center.owner_risks = _command_center__owner_risks


# agent_orchestrator.py
'Controlled planning over authorized task snapshots.\n\nThis module is deliberately side-effect free.  It builds and renders plans;\nthe application layer owns RBAC, approval, execution, and verification through\nthe existing trusted services.\n'
import hashlib as _agent_orchestrator__hashlib
agent_orchestrator.hashlib = _agent_orchestrator__hashlib
import re as _agent_orchestrator__re
agent_orchestrator.re = _agent_orchestrator__re
import time as _agent_orchestrator__time
agent_orchestrator.time = _agent_orchestrator__time
from dataclasses import asdict as _agent_orchestrator__asdict, dataclass as _agent_orchestrator__dataclass, field as _agent_orchestrator__field, replace as _agent_orchestrator__replace
agent_orchestrator.asdict = _agent_orchestrator__asdict
agent_orchestrator.dataclass = _agent_orchestrator__dataclass
agent_orchestrator.field = _agent_orchestrator__field
agent_orchestrator.replace = _agent_orchestrator__replace
from typing import Iterable as _agent_orchestrator__Iterable
agent_orchestrator.Iterable = _agent_orchestrator__Iterable
agent_orchestrator.action_item_sentinel = action_item_sentinel
agent_orchestrator.project_intelligence = project_intelligence
_agent_orchestrator__PLAN_TTL_SECONDS = 1800
agent_orchestrator.PLAN_TTL_SECONDS = _agent_orchestrator__PLAN_TTL_SECONDS
@_agent_orchestrator__dataclass(frozen=True)
class _agent_orchestrator__OrchestratorStep:
    step_id: str
    action_type: str
    target: str
    target_task_id: str | None
    reason: str
    evidence: tuple[str, ...] = ()
    risk_level: str = 'low'
    requires_approval: bool = False
    authorized: bool = False
    executable: bool = False
    status: str = 'proposed'
    execution_result: str | None = None
    task_fingerprint: str | None = None
    target_user_ids: tuple[str, ...] = ()
    prepared_message: str | None = None
agent_orchestrator.OrchestratorStep = _agent_orchestrator__OrchestratorStep
@_agent_orchestrator__dataclass(frozen=True)
class _agent_orchestrator__OrchestratorPlan:
    plan_id: str
    goal: str
    requester_id: str
    created_at: float
    expires_at: float
    status: str
    context: dict
    steps: tuple[agent_orchestrator.OrchestratorStep, ...]
    dependencies: tuple[str, ...] = ()
    expected_outcome: str = 'A reviewed, actionable task plan.'
    validation_result: dict = _agent_orchestrator__field(default_factory=dict)
    approval_status: str = 'not_requested'
    execution_status: str = 'not_started'

    def to_dict(self) -> dict:
        return _agent_orchestrator__asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> 'OrchestratorPlan':
        data = dict(value)
        normalized_steps = []
        for step in data.get('steps', ()):
            step = dict(step)
            step['evidence'] = tuple(step.get('evidence', ()))
            step['target_user_ids'] = tuple(step.get('target_user_ids', ()))
            normalized_steps.append(agent_orchestrator.OrchestratorStep(**step))
        data['steps'] = tuple(normalized_steps)
        data['dependencies'] = tuple(data.get('dependencies', ()))
        return cls(**data)
agent_orchestrator.OrchestratorPlan = _agent_orchestrator__OrchestratorPlan
def _agent_orchestrator___goal_terms(goal: str) -> set[str]:
    ignored = {'prepare', 'everything', 'needed', 'need', 'help', 'team', 'ready', 'current', 'actions', 'action', 'steps', 'next', 'what', 'should', 'before', 'for', 'the', 'our', 'work', 'get', 'make', 'reduce'}
    return {word for word in _agent_orchestrator__re.findall('[a-z0-9]+', goal.casefold()) if len(word) > 2 and word not in ignored}
agent_orchestrator._goal_terms = _agent_orchestrator___goal_terms
def _agent_orchestrator__resolve_goal_context(goal: str) -> str:
    """Map known task concepts to existing deterministic analysis contexts."""
    value = _agent_orchestrator__re.sub('\\s+', ' ', str(goal or '').casefold()).strip(' .?!')
    mappings = (('overdue_work', '\\b(?:current(?:ly)?\\s+)?overdue\\s+(?:work|tasks?|items?)\\b|\\bwhat\\s+is\\s+(?:currently\\s+)?overdue\\b'), ('due_today', '\\b(?:tasks?|items?|work)\\s+due\\s+today\\b|\\bwhat\\s+is\\s+due\\s+today\\b'), ('due_24h', '\\bwithin\\s+(?:the\\s+next\\s+)?24\\s+hours?\\b'), ('due_48h', '\\bwithin\\s+(?:the\\s+next\\s+)?48\\s+hours?\\b'), ('deadline_risk', '\\bdue\\s+soon\\b|\\bapproaching\\s+deadline'), ('priority_p1', '\\b(?:p1|highest[- ]priority|critical)\\s+(?:work|tasks?|items?)\\b'), ('unassigned_tasks', '\\bunassigned\\s+(?:work|tasks?|items?)\\b'), ('current_risks', '\\bcurrent\\s+risks?\\b|\\breduce\\s+(?:the\\s+)?(?:current\\s+)?risks?\\b'), ('team_workload', '\\bteam\\s+workload\\b|\\bcurrent\\s+workload\\b'))
    for context_type, pattern in mappings:
        if _agent_orchestrator__re.search(pattern, value):
            return context_type
    return 'relevant_work'
agent_orchestrator.resolve_goal_context = _agent_orchestrator__resolve_goal_context
def _agent_orchestrator__build_plan(*, goal: str, requester_id: str, tasks: _agent_orchestrator__Iterable[project_intelligence.NormalizedTask], risks: _agent_orchestrator__Iterable[action_item_sentinel.SentinelRisk], fingerprints: dict[str, str], context_type: str='relevant_work', relevant_tasks: _agent_orchestrator__Iterable[project_intelligence.NormalizedTask] | None=None, recommendations: dict[str, object] | None=None, now: float | None=None, ttl_seconds: int=agent_orchestrator.PLAN_TTL_SECONDS) -> agent_orchestrator.OrchestratorPlan:
    """Build a bounded factual plan without authorizing or executing it."""
    now = _agent_orchestrator__time.time() if now is None else now
    task_values = tuple(tasks)
    pending = tuple((task for task in task_values if not task.completed))
    if relevant_tasks is not None:
        relevant = tuple(relevant_tasks)
    else:
        terms = agent_orchestrator._goal_terms(goal)
        relevant = tuple((task for task in pending if terms & set(_agent_orchestrator__re.findall('[a-z0-9]+', task.name.casefold())))) or pending
    relevant_ids = {task.item_id for task in relevant}
    relevant_risks = tuple((risk for risk in risks if any((task_id in relevant_ids for task_id in risk.task_ids))))
    seed = f'{requester_id}|{goal}|{now:.6f}'
    plan_id = _agent_orchestrator__hashlib.sha256(seed.encode()).hexdigest()[:8]
    steps = []
    by_id = {task.item_id: task for task in relevant}
    planned_risk_task_ids = set()
    aggregate_insights = []
    aggregate_seen = set()
    for risk in relevant_risks:
        if len(risk.task_ids) != 1:
            identity = (risk.risk_type, risk.recommendation.casefold())
            if identity not in aggregate_seen:
                aggregate_seen.add(identity)
                aggregate_insights.append({'identity': ':'.join(identity), 'risk_type': risk.risk_type, 'message': 'Multiple urgent action items are creating workload pressure.' if risk.risk_type == 'combined_workload_risk' else risk.recommendation, 'risk_level': risk.severity})
            continue
        if len(planned_risk_task_ids) >= 5:
            continue
        task = by_id.get(risk.task_ids[0])
        if not task:
            continue
        planned_risk_task_ids.add(task.item_id)
        owners = task.owner_ids
        recommendation = (recommendations or {}).get(task.item_id)
        action = getattr(recommendation, 'action_type', None) or ('send_reminder' if owners and risk.risk_type in {'overdue', 'deadline_risk'} else 'review_owner_assignment')
        executable = bool(getattr(recommendation, 'executable', action == 'send_reminder'))
        steps.append(agent_orchestrator.OrchestratorStep(step_id=f'{plan_id}-{len(steps) + 1}', action_type=action, target=task.name, target_task_id=task.item_id, reason=getattr(recommendation, 'recommendation', None) or risk.recommendation, evidence=risk.reasons, risk_level=risk.severity, requires_approval=executable, authorized=False, executable=executable, task_fingerprint=fingerprints.get(task.item_id), target_user_ids=owners, prepared_message=getattr(recommendation, 'prepared_message', None)))
    for task in relevant:
        if task.item_id in planned_risk_task_ids:
            continue
        steps.append(agent_orchestrator.OrchestratorStep(step_id=f'{plan_id}-{len(steps) + 1}', action_type='review_task', target=task.name, target_task_id=task.item_id, reason='Review the current task status.', risk_level='low', authorized=True, executable=False, status='ready', task_fingerprint=fingerprints.get(task.item_id), target_user_ids=task.owner_ids))
    context = {'context_type': context_type, 'task_count': len(task_values), 'pending_count': len(pending), 'relevant_count': len(relevant), 'risk_count': len(relevant_risks), 'relevant_task_ids': sorted(relevant_ids), 'team_insights': aggregate_insights}
    return agent_orchestrator.OrchestratorPlan(plan_id, goal.strip(), requester_id, now, now + ttl_seconds, 'proposed', context, tuple(steps), expected_outcome='Review the relevant risks and execute only approved, current actions.')
agent_orchestrator.build_plan = _agent_orchestrator__build_plan
def _agent_orchestrator__validate_plan(plan: agent_orchestrator.OrchestratorPlan, authorize) -> agent_orchestrator.OrchestratorPlan:
    """Apply deterministic application-owned policy decisions to every step."""
    validated = []
    for step in plan.steps:
        if not step.executable:
            validated.append(_agent_orchestrator__replace(step, authorized=True, status='ready'))
            continue
        allowed, reason = authorize(step)
        validated.append(_agent_orchestrator__replace(step, authorized=bool(allowed), status='ready' if allowed else 'unauthorized', execution_result=None if allowed else reason))
    authorized = sum((step.authorized for step in validated))
    return _agent_orchestrator__replace(plan, steps=tuple(validated), status='validated', validation_result={'authorized': authorized, 'unauthorized': len(validated) - authorized}, approval_status='required' if any((step.requires_approval and step.authorized for step in validated)) else 'not_required')
agent_orchestrator.validate_plan = _agent_orchestrator__validate_plan
def _agent_orchestrator__update_step(plan: agent_orchestrator.OrchestratorPlan, step_id: str, **changes) -> agent_orchestrator.OrchestratorPlan:
    return _agent_orchestrator__replace(plan, steps=tuple((_agent_orchestrator__replace(step, **changes) if step.step_id == step_id else step for step in plan.steps)))
agent_orchestrator.update_step = _agent_orchestrator__update_step


# task_simulation.py
'Deterministic, side-effect-free task scenario modeling.'
import hashlib as _task_simulation__hashlib
task_simulation.hashlib = _task_simulation__hashlib
import json as _task_simulation__json
task_simulation.json = _task_simulation__json
import re as _task_simulation__re
task_simulation.re = _task_simulation__re
import time as _task_simulation__time
task_simulation.time = _task_simulation__time
from dataclasses import asdict as _task_simulation__asdict, dataclass as _task_simulation__dataclass, replace as _task_simulation__replace
task_simulation.asdict = _task_simulation__asdict
task_simulation.dataclass = _task_simulation__dataclass
task_simulation.replace = _task_simulation__replace
from datetime import date as _task_simulation__date, timedelta as _task_simulation__timedelta
task_simulation.date = _task_simulation__date
task_simulation.timedelta = _task_simulation__timedelta
from typing import Iterable as _task_simulation__Iterable
task_simulation.Iterable = _task_simulation__Iterable
task_simulation.action_item_sentinel = action_item_sentinel
task_simulation.project_intelligence = project_intelligence
_task_simulation__SUPPORTED_OPERATIONS = {'assign_task', 'reassign_task', 'change_due_date', 'change_priority', 'complete_task', 'leave_unchanged', 'workload_redistribution'}
task_simulation.SUPPORTED_OPERATIONS = _task_simulation__SUPPORTED_OPERATIONS
@_task_simulation__dataclass(frozen=True)
class _task_simulation__ScenarioRequest:
    operation: str
    goal: str
    task_reference: str | None = None
    assignee_names: tuple[str, ...] = ()
    priority: str | None = None
    due_date: _task_simulation__date | None = None
    due_date_offset_days: int | None = None
    target_unassigned: bool = False
    target_priority: str | None = None
    target_overdue: bool = False
    target_owner_name: str | None = None
    target_owner_self: bool = False
    target_due: str | None = None
    selector_plural: bool = False
    target_assignee: str | None = None
    compare: bool = False
task_simulation.ScenarioRequest = _task_simulation__ScenarioRequest
@_task_simulation__dataclass(frozen=True)
class _task_simulation__SimulationResult:
    scenario_id: str
    fingerprint: str
    requester_id: str
    created_at: float
    expires_at: float
    goal: str
    operation: str
    source_task_ids: tuple[str, ...]
    parameters: dict
    baseline_snapshot_version: str
    baseline_task_fingerprints: dict
    baseline_metrics: dict
    simulated_metrics: dict
    impact: dict
    positive: tuple[str, ...]
    tradeoffs: tuple[str, ...]
    unchanged: tuple[str, ...]
    assumptions: tuple[str, ...]
    confidence: str = 'deterministic'
    status: str = 'simulated'
    decision_id: str | None = None

    def to_dict(self) -> dict:
        return _task_simulation__asdict(self)
task_simulation.SimulationResult = _task_simulation__SimulationResult
def _task_simulation___clean_reference(value: str) -> str:
    value = _task_simulation__re.sub('\\b(?:the|this|that)\\s+task\\b', ' ', value, flags=_task_simulation__re.I)
    value = _task_simulation__re.sub('\\btask\\b', ' ', value, flags=_task_simulation__re.I)
    return _task_simulation__re.sub('\\s+', ' ', value).strip(' .?!"')
task_simulation._clean_reference = _task_simulation___clean_reference
def _task_simulation___semantic_selector(value: str) -> dict:
    """Extract factual task filters while retaining genuine title text."""
    raw = _task_simulation__re.sub('\\s+', ' ', str(value or '')).strip()
    lower = raw.casefold()
    priority = _task_simulation__re.search('\\b(P[1-4])\\b', raw, _task_simulation__re.I)
    owner = _task_simulation__re.search("\\b([A-Za-z][\\w.-]*)['’]s\\s+(?:P[1-4]\\s+)?tasks?\\b", raw, _task_simulation__re.I)
    target_due = None
    if _task_simulation__re.search('\\bdue\\s+today\\b', lower):
        target_due = 'today'
    elif _task_simulation__re.search('\\bdue\\s+tomorrow\\b', lower):
        target_due = 'tomorrow'
    elif _task_simulation__re.search('\\bdue\\s+within\\s+48\\s+hours?\\b', lower):
        target_due = 'within_48h'
    semantic = bool(_task_simulation__re.search('\\bunassigned\\b|\\boverdue\\b|\\bmy\\s+(?:p[1-4]\\s+)?tasks?\\b', lower) or owner or target_due)
    return {'task_reference': None if semantic else task_simulation._clean_reference(raw), 'target_unassigned': bool(_task_simulation__re.search('\\bunassigned\\b', lower)), 'target_priority': priority.group(1).upper() if priority else 'P1' if _task_simulation__re.search('\\b(?:high|highest|urgent|critical)[ -]priority\\b', lower) else None, 'target_overdue': bool(_task_simulation__re.search('\\boverdue\\b', lower)), 'target_owner_name': owner.group(1) if owner else None, 'target_owner_self': bool(_task_simulation__re.search('\\bmy\\s+(?:p[1-4]\\s+)?tasks?\\b', lower)), 'target_due': target_due, 'selector_plural': bool(_task_simulation__re.search('\\b(?:all|every|tasks)\\b', lower))}
task_simulation._semantic_selector = _task_simulation___semantic_selector
def _task_simulation___next_weekday(today: _task_simulation__date, weekday: int) -> _task_simulation__date:
    days = (weekday - today.weekday()) % 7
    return today + _task_simulation__timedelta(days=days or 7)
task_simulation._next_weekday = _task_simulation___next_weekday
def _task_simulation___scenario_date(value: str, today: _task_simulation__date) -> _task_simulation__date | None:
    lower = value.casefold().strip(' .?!')
    if lower == 'today':
        return today
    if lower == 'tomorrow':
        return today + _task_simulation__timedelta(days=1)
    weekdays = {name: index for index, name in enumerate(('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'))}
    for name, weekday in weekdays.items():
        if _task_simulation__re.fullmatch(f'(?:next\\s+)?{name}', lower):
            return task_simulation._next_weekday(today, weekday)
    try:
        return _task_simulation__date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None
task_simulation._scenario_date = _task_simulation___scenario_date
def _task_simulation__parse_request(text: str, today: _task_simulation__date | None=None) -> dict | None:
    """Parse explicit simulation and ledger language without an LLM."""
    today = today or _task_simulation__date.today()
    raw = _task_simulation__re.sub('\\s+', ' ', str(text or '')).strip().rstrip('.?!')
    lower = raw.casefold().rstrip('.?!')
    decision = _task_simulation__re.fullmatch('(?:what happened after|show)\\s+decision\\s+([a-f0-9]{8,16})', lower)
    if decision:
        return {'intent': 'simulation', 'simulation_mode': 'show_decision', 'decision_id': decision.group(1)}
    expected = _task_simulation__re.fullmatch('compare expected (?:vs|versus) actual(?: for)? decision\\s+([a-f0-9]{8,16})', lower)
    if expected:
        return {'intent': 'simulation', 'simulation_mode': 'verify_decision', 'decision_id': expected.group(1)}
    show_one = _task_simulation__re.fullmatch('show simulation\\s+([a-f0-9]{8,16})', lower)
    if show_one:
        return {'intent': 'simulation', 'simulation_mode': 'show_scenario', 'scenario_id': show_one.group(1)}
    if _task_simulation__re.fullmatch('(?:show\\s+)?(?:my\\s+)?(?:recent simulations|decision history|failed decisions)', lower):
        return {'intent': 'simulation', 'simulation_mode': 'history', 'failed_only': 'failed' in lower}
    if _task_simulation__re.fullmatch('(?:why did we choose this|what happened after that decision)', lower):
        return {'intent': 'simulation', 'simulation_mode': 'show_decision'}
    if _task_simulation__re.fullmatch('compare expected (?:vs|versus) actual', lower):
        return {'intent': 'simulation', 'simulation_mode': 'verify_decision'}
    if _task_simulation__re.fullmatch('(?:use|prepare)\\s+(?:this|that|the current)\\s+scenario', lower):
        return {'intent': 'simulation', 'simulation_mode': 'prepare'}
    explicit = bool(_task_simulation__re.match('^(?:what (?:would )?happens? if|what if|simulate|model|compare|what would change|what would improve|what would reduce|hypothetical(?:ly)?|if (?:i|we) (?:move|moved|change|changed))', lower))
    if not explicit:
        return None
    if _task_simulation__re.search('\\b(?:do nothing|nothing changes|leave (?:everything|the current .+?) (?:unchanged|as it is))\\b', lower):
        request = task_simulation.ScenarioRequest('leave_unchanged', raw)
        return {'intent': 'simulation', 'simulation_mode': 'create', 'scenario': _task_simulation__asdict(request)}
    if _task_simulation__re.search('(?:safest way to reduce|what would reduce) (?:our |the )?(?:current )?deadline risk', lower):
        request = task_simulation.ScenarioRequest('workload_redistribution', raw)
        return {'intent': 'simulation', 'simulation_mode': 'create', 'scenario': _task_simulation__asdict(request)}
    compare = _task_simulation__re.search('compare(?: the impact of)? assigning (.+?) to ([A-Za-z][\\w.-]*)\\s+(?:vs|versus|or)\\s+([A-Za-z][\\w.-]*)$', raw, _task_simulation__re.I)
    if compare:
        selector = task_simulation._semantic_selector(compare.group(1))
        request = task_simulation.ScenarioRequest('assign_task', raw, assignee_names=(compare.group(2), compare.group(3)), target_assignee=compare.group(2), compare=True, **selector)
        return {'intent': 'simulation', 'simulation_mode': 'compare', 'scenario': _task_simulation__asdict(request)}
    assign = _task_simulation__re.search('(?:assign(?:ing)?|give|giving)\\s+(.+?)\\s+to\\s+([A-Za-z][\\w.-]*)$', raw, _task_simulation__re.I)
    if assign:
        selector = task_simulation._semantic_selector(assign.group(1))
        request = task_simulation.ScenarioRequest('assign_task', raw, assignee_names=(assign.group(2),), target_assignee=assign.group(2), **selector)
        return {'intent': 'simulation', 'simulation_mode': 'create', 'scenario': _task_simulation__asdict(request)}
    due = _task_simulation__re.search('(?:move|moving|moved)\\s+(.+?)\\s+(?:(?:deadline|due date)\\s+)?to\\s+(.+?)(?:\\s*,?\\s*what would happen)?$', raw, _task_simulation__re.I)
    if not due:
        due = _task_simulation__re.search('(.+?)\\s+(?:are|is)\\s+moved\\s+to\\s+(.+?)(?:\\s*,?\\s*what would happen)?$', raw, _task_simulation__re.I)
    if not due:
        due = _task_simulation__re.search('(?:if\\s+)?(.+?)\\s+is\\s+due\\s+(.+)$', raw, _task_simulation__re.I)
    if due and (parsed_date := task_simulation._scenario_date(due.group(2), today)):
        selector = task_simulation._semantic_selector(due.group(1))
        request = task_simulation.ScenarioRequest('change_due_date', raw, due_date=parsed_date, **selector)
        return {'intent': 'simulation', 'simulation_mode': 'create', 'scenario': _task_simulation__asdict(request)}
    shifted = _task_simulation__re.search('(?:move|moving|moved)\\s+(.+?)\\s+by\\s+(?:one|1)\\s+week(?:\\s*,?\\s*what would happen)?$', raw, _task_simulation__re.I)
    if shifted:
        selector = task_simulation._semantic_selector(shifted.group(1))
        request = task_simulation.ScenarioRequest('change_due_date', raw, due_date_offset_days=7, **selector)
        return {'intent': 'simulation', 'simulation_mode': 'create', 'scenario': _task_simulation__asdict(request)}
    priority = _task_simulation__re.search('(?:make|making)\\s+(.+?)\\s+(P[1-4])$', raw, _task_simulation__re.I)
    if priority:
        selector = task_simulation._semantic_selector(priority.group(1))
        request = task_simulation.ScenarioRequest('change_priority', raw, priority=priority.group(2).upper(), **selector)
        return {'intent': 'simulation', 'simulation_mode': 'create', 'scenario': _task_simulation__asdict(request)}
    complete = _task_simulation__re.search('(?:complete|completing|finish|finishing)\\s+(.+)$', raw, _task_simulation__re.I)
    if complete:
        selector = task_simulation._semantic_selector(complete.group(1))
        request = task_simulation.ScenarioRequest('complete_task', raw, **selector)
        return {'intent': 'simulation', 'simulation_mode': 'create', 'scenario': _task_simulation__asdict(request)}
    if lower.startswith('compare'):
        return None
    return {'intent': 'clarify', 'clarification': 'Please specify the task and hypothetical assignment, deadline, priority, or completion change.'}
task_simulation.parse_request = _task_simulation__parse_request
def _task_simulation__snapshot_version(tasks: _task_simulation__Iterable[project_intelligence.NormalizedTask]) -> str:
    state = [{'id': task.item_id, 'owners': task.owner_ids, 'priority': task.priority, 'due': task.due_date.isoformat() if task.due_date else None, 'completed': task.completed, 'name': task.name} for task in tasks]
    return _task_simulation__hashlib.sha256(_task_simulation__json.dumps(state, sort_keys=True).encode()).hexdigest()
task_simulation.snapshot_version = _task_simulation__snapshot_version
def _task_simulation__task_state_fingerprint(task: project_intelligence.NormalizedTask) -> str:
    return _task_simulation__hashlib.sha256(_task_simulation__json.dumps({'id': task.item_id, 'owners': task.owner_ids, 'priority': task.priority, 'due': task.due_date.isoformat() if task.due_date else None, 'completed': task.completed, 'name': task.name}, sort_keys=True).encode()).hexdigest()
task_simulation.task_state_fingerprint = _task_simulation__task_state_fingerprint
def _task_simulation__calculate_metrics(tasks, today: _task_simulation__date) -> dict:
    pending = [task for task in tasks if not task.completed]
    owners = {}
    for task in pending:
        for owner in task.owner_ids:
            row = owners.setdefault(owner, {'pending': 0, 'p1': 0, 'overdue': 0, 'due_soon': 0})
            row['pending'] += 1
            row['p1'] += task.priority == 'P1'
            row['overdue'] += bool(task.due_date and task.due_date < today)
            row['due_soon'] += bool(task.due_date and today <= task.due_date <= today + _task_simulation__timedelta(days=2))
    priorities = {key: sum((task.priority == key for task in pending)) for key in ('P1', 'P2', 'P3', 'P4')}
    return {'pending': len(pending), 'completed': len(tasks) - len(pending), 'unassigned': sum((not task.owner_ids for task in pending)), 'overdue': sum((bool(task.due_date and task.due_date < today) for task in pending)), 'due_today': sum((task.due_date == today for task in pending)), 'due_24h': sum((bool(task.due_date and today <= task.due_date <= today + _task_simulation__timedelta(days=1)) for task in pending)), 'due_48h': sum((bool(task.due_date and today <= task.due_date <= today + _task_simulation__timedelta(days=2)) for task in pending)), 'due_this_week': sum((bool(task.due_date and today <= task.due_date <= today + _task_simulation__timedelta(days=7)) for task in pending)), 'priorities': priorities, 'owners': owners}
task_simulation.calculate_metrics = _task_simulation__calculate_metrics
def _task_simulation___risk_keys(tasks, today):
    settings = action_item_sentinel.SentinelSettings.from_env()
    risks = action_item_sentinel.detect_risks(tasks, today, settings.warning_days, settings.combined_task_threshold)
    return {f'{risk.risk_type}:{','.join(sorted(risk.task_ids))}' for risk in risks}
task_simulation._risk_keys = _task_simulation___risk_keys
def _task_simulation__project(tasks, operation: str, task_ids: _task_simulation__Iterable[str], parameters: dict):
    """Clone normalized tasks and apply a hypothetical change only to the clone."""
    wanted = set(task_ids)
    projected = []
    for task in tasks:
        clone = _task_simulation__replace(task)
        if task.item_id in wanted:
            if operation in {'assign_task', 'reassign_task', 'workload_redistribution'}:
                clone = _task_simulation__replace(clone, owner_ids=tuple(parameters.get('assignee_ids') or ()))
            elif operation == 'change_due_date':
                if parameters.get('due_date_offset_days') is not None:
                    clone = _task_simulation__replace(clone, due_date=clone.due_date + _task_simulation__timedelta(days=int(parameters['due_date_offset_days'])) if clone.due_date else None)
                else:
                    clone = _task_simulation__replace(clone, due_date=_task_simulation__date.fromisoformat(parameters['due_date']))
            elif operation == 'change_priority':
                clone = _task_simulation__replace(clone, priority=parameters['priority'])
            elif operation == 'complete_task':
                clone = _task_simulation__replace(clone, completed=True, status='Completed')
        projected.append(clone)
    return projected
task_simulation.project = _task_simulation__project
def _task_simulation__simulate(*, requester_id: str, goal: str, operation: str, tasks, task_ids: _task_simulation__Iterable[str], parameters: dict, now: float | None=None, today: _task_simulation__date | None=None, ttl_seconds: int=1800) -> task_simulation.SimulationResult:
    if operation not in task_simulation.SUPPORTED_OPERATIONS:
        raise ValueError('That simulation operation is not supported.')
    now, today = (_task_simulation__time.time() if now is None else now, today or _task_simulation__date.today())
    tasks, task_ids = (list(tasks), tuple(task_ids))
    projected = task_simulation.project(tasks, operation, task_ids, parameters)
    baseline, modeled = (task_simulation.calculate_metrics(tasks, today), task_simulation.calculate_metrics(projected, today))
    baseline_risks, modeled_risks = (task_simulation._risk_keys(tasks, today), task_simulation._risk_keys(projected, today))
    impact = {'pending_delta': modeled['pending'] - baseline['pending'], 'completed_delta': modeled['completed'] - baseline['completed'], 'unassigned_delta': modeled['unassigned'] - baseline['unassigned'], 'overdue_delta': modeled['overdue'] - baseline['overdue'], 'risk_removed': sorted(baseline_risks - modeled_risks), 'risk_introduced': sorted(modeled_risks - baseline_risks), 'risk_unchanged': sorted(baseline_risks & modeled_risks)}
    owner_id = next(iter(parameters.get('assignee_ids') or ()), None)
    before_owner = baseline['owners'].get(owner_id, {}) if owner_id else {}
    after_owner = modeled['owners'].get(owner_id, {}) if owner_id else {}
    impact['owner_delta'] = {key: after_owner.get(key, 0) - before_owner.get(key, 0) for key in ('pending', 'p1', 'overdue', 'due_soon')}
    positive, negative, unchanged = ([], [], [])
    if impact['unassigned_delta'] < 0:
        positive.append(f'Removes {-impact['unassigned_delta']} unassigned task' + ('s' if impact['unassigned_delta'] != -1 else ''))
    if impact['completed_delta'] > 0:
        positive.append(f'Models {impact['completed_delta']} additional completed task' + ('s' if impact['completed_delta'] != 1 else ''))
    if impact['overdue_delta'] < 0:
        positive.append(f'Reduces overdue work by {-impact['overdue_delta']}')
    if impact['risk_removed']:
        positive.append(f'Removes {len(impact['risk_removed'])} current risk signal' + ('s' if len(impact['risk_removed']) != 1 else ''))
    if impact['owner_delta'].get('pending', 0) > 0:
        negative.append("The selected owner's pending workload increases")
    if impact['owner_delta'].get('p1', 0) > 0:
        negative.append("The selected owner's P1 concentration increases")
    if impact['risk_introduced']:
        negative.append(f'Introduces {len(impact['risk_introduced'])} modeled risk signal' + ('s' if len(impact['risk_introduced']) != 1 else ''))
    for key, label in (('overdue_delta', 'Overdue count'), ('pending_delta', 'Total pending count')):
        if impact[key] == 0:
            unchanged.append(f'{label} remains unchanged')
    if operation == 'leave_unchanged':
        unchanged = ['Current unresolved exposure remains unchanged']
    baseline_version = task_simulation.snapshot_version(tasks)
    seed = _task_simulation__json.dumps({'requester': requester_id, 'operation': operation, 'tasks': task_ids, 'parameters': parameters, 'baseline': baseline_version}, sort_keys=True)
    fingerprint = _task_simulation__hashlib.sha256(seed.encode()).hexdigest()
    return task_simulation.SimulationResult(scenario_id=fingerprint[:10], fingerprint=fingerprint, requester_id=requester_id, created_at=now, expires_at=now + ttl_seconds, goal=goal, operation=operation, source_task_ids=task_ids, parameters=parameters, baseline_snapshot_version=baseline_version, baseline_task_fingerprints={task.item_id: task_simulation.task_state_fingerprint(task) for task in tasks if task.item_id in set(task_ids)}, baseline_metrics=baseline, simulated_metrics=modeled, impact=impact, positive=tuple(positive), tradeoffs=tuple(negative), unchanged=tuple(unchanged), assumptions=('Only the selected fields change; all other task state remains constant.', 'Metrics are projected from the current authorized Slack List snapshot.'))
task_simulation.simulate = _task_simulation__simulate
def _task_simulation__is_stale(result: task_simulation.SimulationResult, current_tasks) -> bool:
    current = {task.item_id: task for task in current_tasks}
    return any((task_id not in current or task_simulation.task_state_fingerprint(current[task_id]) != fingerprint for task_id, fingerprint in result.baseline_task_fingerprints.items()))
task_simulation.is_stale = _task_simulation__is_stale


# decision_ledger.py
'Persistent, idempotent decision records for task simulations.'
import json as _decision_ledger__json
decision_ledger.json = _decision_ledger__json
import hashlib as _decision_ledger__hashlib
decision_ledger.hashlib = _decision_ledger__hashlib
import sqlite3 as _decision_ledger__sqlite3
decision_ledger.sqlite3 = _decision_ledger__sqlite3
import time as _decision_ledger__time
decision_ledger.time = _decision_ledger__time
from dataclasses import replace as _decision_ledger__replace
decision_ledger.replace = _decision_ledger__replace
class _decision_ledger__DecisionLedger:

    def __init__(self, path: str):
        self.path = path
        with self._connect() as connection:
            connection.execute('\n                CREATE TABLE IF NOT EXISTS decision_ledger (\n                    decision_id TEXT PRIMARY KEY,\n                    scenario_id TEXT NOT NULL,\n                    fingerprint TEXT NOT NULL UNIQUE,\n                    requester_id TEXT NOT NULL,\n                    list_id TEXT NOT NULL,\n                    created_at REAL NOT NULL,\n                    scenario_json TEXT NOT NULL,\n                    approval_status TEXT NOT NULL,\n                    execution_status TEXT NOT NULL,\n                    verification_status TEXT NOT NULL,\n                    actual_json TEXT,\n                    plan_id TEXT\n                )\n            ')

    def _connect(self):
        return _decision_ledger__sqlite3.connect(self.path, timeout=10)

    def create(self, result: task_simulation.SimulationResult, list_id: str) -> task_simulation.SimulationResult:
        scoped_fingerprint = _decision_ledger__hashlib.sha256(f'{list_id}|{result.fingerprint}'.encode()).hexdigest()
        decision_id = scoped_fingerprint[:12]
        frozen = _decision_ledger__replace(result, decision_id=decision_id)
        with self._connect() as connection:
            connection.execute('INSERT OR IGNORE INTO decision_ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (decision_id, result.scenario_id, scoped_fingerprint, result.requester_id, list_id, result.created_at, _decision_ledger__json.dumps(frozen.to_dict(), default=str), 'not_requested', 'not_started', 'not_verified', None, None))
            row = connection.execute('SELECT scenario_json FROM decision_ledger WHERE fingerprint=?', (scoped_fingerprint,)).fetchone()
        return self._result(_decision_ledger__json.loads(row[0]))

    @staticmethod
    def _result(value: dict) -> task_simulation.SimulationResult:
        for name in ('source_task_ids', 'positive', 'tradeoffs', 'unchanged', 'assumptions'):
            value[name] = tuple(value.get(name) or ())
        return task_simulation.SimulationResult(**value)

    def get(self, decision_id: str, requester_id: str, list_id: str):
        with self._connect() as connection:
            row = connection.execute('SELECT scenario_json, approval_status, execution_status, verification_status, actual_json, plan_id FROM decision_ledger WHERE decision_id=? AND requester_id=? AND list_id=?', (decision_id, requester_id, list_id)).fetchone()
        if not row:
            return None
        return {'scenario': self._result(_decision_ledger__json.loads(row[0])), 'approval_status': row[1], 'execution_status': row[2], 'verification_status': row[3], 'actual': _decision_ledger__json.loads(row[4]) if row[4] else None, 'plan_id': row[5]}

    def get_scenario(self, scenario_id: str, requester_id: str, list_id: str):
        with self._connect() as connection:
            row = connection.execute('SELECT decision_id FROM decision_ledger WHERE scenario_id=? AND requester_id=? AND list_id=? ORDER BY created_at DESC LIMIT 1', (scenario_id, requester_id, list_id)).fetchone()
        return self.get(row[0], requester_id, list_id) if row else None

    def recent(self, requester_id: str, list_id: str, limit: int=10, failed_only: bool=False):
        where = " AND (execution_status='failed' OR verification_status='variance')" if failed_only else ''
        with self._connect() as connection:
            rows = connection.execute(f'SELECT decision_id, scenario_id, created_at, scenario_json, execution_status, verification_status FROM decision_ledger WHERE requester_id=? AND list_id=?{where} ORDER BY created_at DESC LIMIT ?', (requester_id, list_id, limit)).fetchall()
        return [{'decision_id': row[0], 'scenario_id': row[1], 'created_at': row[2], 'scenario': self._result(_decision_ledger__json.loads(row[3])), 'execution_status': row[4], 'verification_status': row[5]} for row in rows]

    def mark_prepared(self, decision_id: str, requester_id: str, list_id: str, plan_id: str):
        with self._connect() as connection:
            connection.execute("UPDATE decision_ledger SET approval_status='required', plan_id=? WHERE decision_id=? AND requester_id=? AND list_id=?", (plan_id, decision_id, requester_id, list_id))

    def record_outcome(self, decision_id: str, requester_id: str, list_id: str, actual: dict, verified: bool):
        with self._connect() as connection:
            connection.execute("UPDATE decision_ledger SET approval_status='approved', execution_status='executed', verification_status=?, actual_json=? WHERE decision_id=? AND requester_id=? AND list_id=?", ('verified' if verified else 'variance', _decision_ledger__json.dumps(actual, default=str), decision_id, requester_id, list_id))
decision_ledger.DecisionLedger = _decision_ledger__DecisionLedger


# deadline_reminders.py
'Read-only deadline reminders for Slack List action items.'
import logging as _deadline_reminders__logging
deadline_reminders.logging = _deadline_reminders__logging
import os as _deadline_reminders__os
deadline_reminders.os = _deadline_reminders__os
import sqlite3 as _deadline_reminders__sqlite3
deadline_reminders.sqlite3 = _deadline_reminders__sqlite3
import threading as _deadline_reminders__threading
deadline_reminders.threading = _deadline_reminders__threading
import hashlib as _deadline_reminders__hashlib
deadline_reminders.hashlib = _deadline_reminders__hashlib
import json as _deadline_reminders__json
deadline_reminders.json = _deadline_reminders__json
from dataclasses import dataclass as _deadline_reminders__dataclass
deadline_reminders.dataclass = _deadline_reminders__dataclass
from datetime import date as _deadline_reminders__date, datetime as _deadline_reminders__datetime, timedelta as _deadline_reminders__timedelta
deadline_reminders.date = _deadline_reminders__date
deadline_reminders.datetime = _deadline_reminders__datetime
deadline_reminders.timedelta = _deadline_reminders__timedelta
from html import escape as _deadline_reminders__escape
deadline_reminders.escape = _deadline_reminders__escape
from pathlib import Path as _deadline_reminders__Path
deadline_reminders.Path = _deadline_reminders__Path
from typing import Callable as _deadline_reminders__Callable, Iterable as _deadline_reminders__Iterable
deadline_reminders.Callable = _deadline_reminders__Callable
deadline_reminders.Iterable = _deadline_reminders__Iterable
from zoneinfo import ZoneInfo as _deadline_reminders__ZoneInfo
deadline_reminders.ZoneInfo = _deadline_reminders__ZoneInfo
_deadline_reminders__logger = _deadline_reminders__logging.getLogger('slack_list.reminders')
deadline_reminders.logger = _deadline_reminders__logger
def _deadline_reminders___env_bool(name: str, default: bool) -> bool:
    value = _deadline_reminders__os.getenv(name)
    return default if value is None else value.strip().casefold() in {'1', 'true', 'yes', 'on'}
deadline_reminders._env_bool = _deadline_reminders___env_bool
def _deadline_reminders___env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(_deadline_reminders__os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))
deadline_reminders._env_int = _deadline_reminders___env_int
@_deadline_reminders__dataclass(frozen=True)
class _deadline_reminders__ReminderSettings:
    enabled: bool = True
    hour: int = 9
    minute: int = 0
    timezone: str = 'Asia/Kathmandu'
    due_today: bool = True
    due_tomorrow: bool = True
    overdue: bool = True
    scan_interval_seconds: int = 3600
    interval_enabled: bool = False
    repeat_hours: int = 24
    escalation_days: int = 3
    high_priority_days: int = 3
    notification_behavior: str = 'direct'
    due_soon_hours: int = 24
    fallback_channel: str = ''
    fallback_actor_id: str = ''
    workspace_timezone: str = ''
    application_timezone_configured: bool = False

    @classmethod
    def from_env(cls) -> 'ReminderSettings':
        configured_timezone = _deadline_reminders__os.getenv('DEADLINE_REMINDER_TIMEZONE')
        timezone = (configured_timezone or 'Asia/Kathmandu').strip()
        try:
            _deadline_reminders__ZoneInfo(timezone)
        except Exception:
            deadline_reminders.logger.warning('scheduler_error stage=config reason=invalid_timezone fallback=Asia/Kathmandu')
            timezone = 'Asia/Kathmandu'
        workspace_timezone = _deadline_reminders__os.getenv('SLACK_WORKSPACE_TIMEZONE', '').strip()
        if workspace_timezone:
            try:
                _deadline_reminders__ZoneInfo(workspace_timezone)
            except Exception:
                deadline_reminders.logger.warning('scheduler_error stage=config reason=invalid_workspace_timezone')
                workspace_timezone = ''
        behavior = _deadline_reminders__os.getenv('FOLLOWUP_NOTIFICATION_BEHAVIOR', 'direct').strip().casefold() or 'direct'
        if behavior not in {'direct', 'disabled'}:
            deadline_reminders.logger.warning('scheduler_error stage=config reason=invalid_notification_behavior fallback=direct')
            behavior = 'direct'
        enabled = deadline_reminders._env_bool('FOLLOW_UP_ENABLED', deadline_reminders._env_bool('DEADLINE_REMINDERS_ENABLED', True)) and behavior != 'disabled'
        interval_minutes = _deadline_reminders__os.getenv('FOLLOW_UP_INTERVAL_MINUTES')
        legacy_interval = _deadline_reminders__os.getenv('FOLLOWUP_SCAN_INTERVAL_SECONDS')
        interval_enabled = interval_minutes is not None or legacy_interval is not None
        interval = deadline_reminders._env_int('FOLLOW_UP_INTERVAL_MINUTES', 60, 1, 1440) * 60 if interval_minutes is not None else deadline_reminders._env_int('FOLLOWUP_SCAN_INTERVAL_SECONDS', 3600, 60, 86400)
        return cls(enabled=enabled, hour=deadline_reminders._env_int('DEADLINE_REMINDER_HOUR', 9, 0, 23), minute=deadline_reminders._env_int('DEADLINE_REMINDER_MINUTE', 0, 0, 59), timezone=timezone, due_today=deadline_reminders._env_bool('DEADLINE_REMINDER_DUE_TODAY', True), due_tomorrow=deadline_reminders._env_bool('DEADLINE_REMINDER_DUE_TOMORROW', True), overdue=deadline_reminders._env_bool('FOLLOW_UP_OVERDUE_ENABLED', deadline_reminders._env_bool('DEADLINE_REMINDER_OVERDUE', True)), scan_interval_seconds=interval, interval_enabled=interval_enabled, repeat_hours=deadline_reminders._env_int('FOLLOWUP_REPEAT_HOURS', 24, 1, 720), escalation_days=deadline_reminders._env_int('FOLLOWUP_ESCALATION_DAYS', 3, 1, 365), high_priority_days=deadline_reminders._env_int('FOLLOWUP_HIGH_PRIORITY_DAYS', 3, 2, 30), notification_behavior=behavior, due_soon_hours=deadline_reminders._env_int('FOLLOW_UP_DUE_SOON_HOURS', 24, 0, 720), fallback_channel=_deadline_reminders__os.getenv('FOLLOW_UP_CHANNEL', '').strip(), fallback_actor_id=_deadline_reminders__os.getenv('FOLLOW_UP_FALLBACK_ACTOR_ID', '').strip(), workspace_timezone=workspace_timezone, application_timezone_configured=configured_timezone is not None)
deadline_reminders.ReminderSettings = _deadline_reminders__ReminderSettings
_deadline_reminders___WEEKDAYS = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
deadline_reminders._WEEKDAYS = _deadline_reminders___WEEKDAYS
@_deadline_reminders__dataclass(frozen=True)
class _deadline_reminders__WeeklySummarySettings:
    enabled: bool = False
    day: int = 4
    hour: int = 17
    minute: int = 0
    channel: str = ''
    actor_id: str = ''

    @classmethod
    def from_env(cls) -> 'WeeklySummarySettings':
        raw_day = _deadline_reminders__os.getenv('WEEKLY_SUMMARY_DAY', 'friday').strip().casefold()
        try:
            day = int(raw_day)
        except ValueError:
            day = deadline_reminders._WEEKDAYS.get(raw_day, 4)
        day = min(6, max(0, day))
        return cls(enabled=deadline_reminders._env_bool('WEEKLY_SUMMARY_ENABLED', False), day=day, hour=deadline_reminders._env_int('WEEKLY_SUMMARY_HOUR', 17, 0, 23), minute=deadline_reminders._env_int('WEEKLY_SUMMARY_MINUTE', 0, 0, 59), channel=_deadline_reminders__os.getenv('WEEKLY_SUMMARY_CHANNEL', '').strip(), actor_id=_deadline_reminders__os.getenv('WEEKLY_SUMMARY_ACTOR_ID', '').strip())
deadline_reminders.WeeklySummarySettings = _deadline_reminders__WeeklySummarySettings
@_deadline_reminders__dataclass(frozen=True)
class _deadline_reminders__WeeklySummaryDelivery:
    list_id: str
    channel_id: str
    period_start: _deadline_reminders__date
    period_end: _deadline_reminders__date
    message: str
deadline_reminders.WeeklySummaryDelivery = _deadline_reminders__WeeklySummaryDelivery
@_deadline_reminders__dataclass(frozen=True)
class _deadline_reminders__ReminderTask:
    task_id: str
    name: str
    owner_id: str
    owner_name: str
    due_date: _deadline_reminders__date
    priority: str | None = None
    completed: bool = False
    status: str | None = None
    timezone: str = 'UTC'
deadline_reminders.ReminderTask = _deadline_reminders__ReminderTask
class _deadline_reminders__ReminderStore:
    """Persistent once-per-task/recipient/day delivery ledger."""

    def __init__(self, path: str):
        self.path = str(_deadline_reminders__Path(path))
        self._lock = _deadline_reminders__threading.Lock()
        with self._connect() as connection:
            connection.execute('\n                CREATE TABLE IF NOT EXISTS deadline_reminders (\n                    task_id TEXT NOT NULL,\n                    recipient_id TEXT NOT NULL,\n                    reminder_date TEXT NOT NULL,\n                    category TEXT NOT NULL,\n                    sent_at TEXT NOT NULL,\n                    PRIMARY KEY (task_id, recipient_id, reminder_date)\n                )\n            ')
            connection.execute('\n                CREATE TABLE IF NOT EXISTS weekly_summary_runs (\n                    list_id TEXT NOT NULL,\n                    channel_id TEXT NOT NULL,\n                    period_start TEXT NOT NULL,\n                    period_end TEXT NOT NULL,\n                    sent_at REAL,\n                    lease_until REAL NOT NULL DEFAULT 0,\n                    PRIMARY KEY (list_id, channel_id, period_start, period_end)\n                )\n            ')
            connection.execute('\n                CREATE TABLE IF NOT EXISTS followup_notifications (\n                    task_id TEXT NOT NULL,\n                    recipient_id TEXT NOT NULL,\n                    condition TEXT NOT NULL,\n                    state_hash TEXT NOT NULL,\n                    last_sent REAL,\n                    lease_until REAL NOT NULL DEFAULT 0,\n                    PRIMARY KEY (task_id, recipient_id, condition)\n                )\n            ')

    def _connect(self):
        return _deadline_reminders__sqlite3.connect(self.path, timeout=10)

    def was_sent(self, task_id: str, recipient_id: str, reminder_date: _deadline_reminders__date) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute('SELECT 1 FROM deadline_reminders WHERE task_id=? AND recipient_id=? AND reminder_date=?', (task_id, recipient_id, reminder_date.isoformat())).fetchone()
        return bool(row)

    def mark_sent(self, task_id: str, recipient_id: str, reminder_date: _deadline_reminders__date, category: str, sent_at: _deadline_reminders__datetime) -> None:
        with self._lock, self._connect() as connection:
            connection.execute('INSERT OR IGNORE INTO deadline_reminders VALUES (?, ?, ?, ?, ?)', (task_id, recipient_id, reminder_date.isoformat(), category, sent_at.isoformat()))

    def claim(self, task: deadline_reminders.ReminderTask, condition: str, now: _deadline_reminders__datetime, repeat_hours: int, lease_seconds: int=300) -> bool:
        """Atomically lease one notification across scheduler instances."""
        state_hash = _deadline_reminders__hashlib.sha256(_deadline_reminders__json.dumps({'due': task.due_date.isoformat(), 'priority': task.priority, 'status': task.status, 'completed': task.completed}, sort_keys=True).encode()).hexdigest()
        timestamp = now.timestamp()
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT state_hash, last_sent, lease_until FROM followup_notifications WHERE task_id=? AND recipient_id=? AND condition=?', (task.task_id, task.owner_id, condition)).fetchone()
            if row and row[2] > timestamp:
                return False
            if row and row[0] == state_hash and (row[1] is not None) and (timestamp - row[1] < repeat_hours * 3600):
                return False
            connection.execute('INSERT INTO followup_notifications VALUES (?, ?, ?, ?, NULL, ?) ON CONFLICT(task_id, recipient_id, condition) DO UPDATE SET state_hash=excluded.state_hash, lease_until=excluded.lease_until', (task.task_id, task.owner_id, condition, state_hash, timestamp + lease_seconds))
        return True

    def delivered(self, task: deadline_reminders.ReminderTask, condition: str, now: _deadline_reminders__datetime) -> None:
        with self._lock, self._connect() as connection:
            connection.execute('UPDATE followup_notifications SET last_sent=?, lease_until=0 WHERE task_id=? AND recipient_id=? AND condition=?', (now.timestamp(), task.task_id, task.owner_id, condition))

    def release(self, task: deadline_reminders.ReminderTask, condition: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute('UPDATE followup_notifications SET last_sent=NULL, lease_until=0 WHERE task_id=? AND recipient_id=? AND condition=?', (task.task_id, task.owner_id, condition))

    def claim_weekly(self, delivery: deadline_reminders.WeeklySummaryDelivery, now: _deadline_reminders__datetime, lease_seconds: int=300) -> bool:
        """Atomically lease one channel/period summary across app instances."""
        timestamp = now.timestamp()
        key = (delivery.list_id, delivery.channel_id, delivery.period_start.isoformat(), delivery.period_end.isoformat())
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT sent_at, lease_until FROM weekly_summary_runs WHERE list_id=? AND channel_id=? AND period_start=? AND period_end=?', key).fetchone()
            if row and (row[0] is not None or row[1] > timestamp):
                return False
            connection.execute('INSERT INTO weekly_summary_runs VALUES (?, ?, ?, ?, NULL, ?) ON CONFLICT(list_id, channel_id, period_start, period_end) DO UPDATE SET lease_until=excluded.lease_until', (*key, timestamp + lease_seconds))
        return True

    def weekly_delivered(self, delivery: deadline_reminders.WeeklySummaryDelivery, now: _deadline_reminders__datetime) -> None:
        with self._lock, self._connect() as connection:
            connection.execute('UPDATE weekly_summary_runs SET sent_at=?, lease_until=0 WHERE list_id=? AND channel_id=? AND period_start=? AND period_end=?', (now.timestamp(), delivery.list_id, delivery.channel_id, delivery.period_start.isoformat(), delivery.period_end.isoformat()))

    def release_weekly(self, delivery: deadline_reminders.WeeklySummaryDelivery) -> None:
        with self._lock, self._connect() as connection:
            connection.execute('UPDATE weekly_summary_runs SET lease_until=0 WHERE list_id=? AND channel_id=? AND period_start=? AND period_end=?', (delivery.list_id, delivery.channel_id, delivery.period_start.isoformat(), delivery.period_end.isoformat()))
deadline_reminders.ReminderStore = _deadline_reminders__ReminderStore
def _deadline_reminders___task_line(task: deadline_reminders.ReminderTask, category: str, today: _deadline_reminders__date) -> str:
    values = [_deadline_reminders__escape(task.name, quote=False)]
    if task.priority:
        values.append(_deadline_reminders__escape(task.priority, quote=False))
    values.append(_deadline_reminders__escape(task.owner_name, quote=False))
    if category in {'overdue', 'escalated', 'high_priority_overdue'}:
        days = (today - task.due_date).days
        values.append(f'{days} day{('s' if days != 1 else '')} overdue')
    elif category in {'high_priority', 'due_soon'}:
        days = (task.due_date - today).days
        values.append(f'Due in {days} days')
    elif task.due_date == today:
        values.append('Due today')
    else:
        values.append('Due tomorrow')
    return '• ' + ' — '.join(values)
deadline_reminders._task_line = _deadline_reminders___task_line
def _deadline_reminders__format_reminder(tasks: _deadline_reminders__Iterable[deadline_reminders.ReminderTask], category: str, today: _deadline_reminders__date) -> str:
    tasks = list(tasks)
    if len(tasks) == 1:
        task = tasks[0]
        due = task.due_date.strftime('%b %d').replace(' 0', ' ')
        from src.prompts import REMINDER_FOLLOWUP_PROMPTS
        prompt = REMINDER_FOLLOWUP_PROMPTS.get(category, REMINDER_FOLLOWUP_PROMPTS['deadline'])
        fields = [f'*Task:* {_deadline_reminders__escape(task.name, quote=False)}', f'*Owner:* {_deadline_reminders__escape(task.owner_name, quote=False)}']
        if task.priority:
            fields.append(f'*Priority:* {_deadline_reminders__escape(task.priority, quote=False)}')
        fields.append(f'*Due:* {due}')
        return '🔔 *Action Item Follow-up*\n\n' + '\n'.join(fields) + f'\n\n{prompt}'
    title = {'overdue': '⚠️ Follow-up · Overdue', 'high_priority_overdue': '🚨 Follow-up · P1 Overdue', 'escalated': '🚨 Escalated Follow-up', 'high_priority': '🔔 High-priority Deadline', 'due_soon': '🔔 Follow-up · Due Soon'}.get(category, '🔔 Follow-up · Deadline')
    return f'*{title}*\n\n' + '\n'.join((deadline_reminders._task_line(task, category, today) for task in tasks))
deadline_reminders.format_reminder = _deadline_reminders__format_reminder
class _deadline_reminders__DeadlineReminderScheduler:
    """Daily, non-blocking scheduler with explicit graceful shutdown."""

    def __init__(self, settings: deadline_reminders.ReminderSettings, store: deadline_reminders.ReminderStore, load_tasks: _deadline_reminders__Callable[[], _deadline_reminders__Iterable[deadline_reminders.ReminderTask]], send: _deadline_reminders__Callable[[str, str], None], now: _deadline_reminders__Callable[[], _deadline_reminders__datetime] | None=None, weekly_settings: deadline_reminders.WeeklySummarySettings | None=None, build_weekly: _deadline_reminders__Callable[[_deadline_reminders__date, _deadline_reminders__date], deadline_reminders.WeeklySummaryDelivery | None] | None=None, send_weekly: _deadline_reminders__Callable[[str, str], None] | None=None):
        self.settings = settings
        self.store = store
        self.load_tasks = load_tasks
        self.send = send
        self.now = now or (lambda: _deadline_reminders__datetime.now(_deadline_reminders__ZoneInfo(settings.timezone)))
        self.weekly_settings = weekly_settings or deadline_reminders.WeeklySummarySettings()
        self.build_weekly = build_weekly
        self.send_weekly = send_weekly
        self._stop = _deadline_reminders__threading.Event()
        self._thread: _deadline_reminders__threading.Thread | None = None

    def start(self) -> bool:
        if not self.settings.enabled and (not self.weekly_settings.enabled):
            deadline_reminders.logger.info('reminder_scheduler_disabled')
            return False
        if self._thread and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = _deadline_reminders__threading.Thread(target=self._run, name='slack-deadline-reminders', daemon=True)
        self._thread.start()
        if self.settings.interval_enabled:
            deadline_reminders.logger.info('reminder_scheduler_started mode=interval interval_minutes=%d production_hour=%d production_minute=%d timezone=%s', self.settings.scan_interval_seconds // 60, self.settings.hour, self.settings.minute, self.settings.timezone)
        else:
            deadline_reminders.logger.info('reminder_scheduler_started mode=owner_local hour=%d minute=%d poll_minutes=%d fallback_timezone=%s', self.settings.hour, self.settings.minute, self.settings.scan_interval_seconds // 60, self.settings.timezone)
        return True

    def stop(self, timeout: float=5.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
        deadline_reminders.logger.info('reminder_scheduler_stopped')

    def _next_run(self, now: _deadline_reminders__datetime) -> _deadline_reminders__datetime:
        target = now.replace(hour=self.settings.hour, minute=self.settings.minute, second=0, microsecond=0)
        reminder_target = target if target > now else target + _deadline_reminders__timedelta(days=1)
        weekly = self.weekly_settings
        if not weekly.enabled:
            return reminder_target
        days = (weekly.day - now.weekday()) % 7
        weekly_target = (now + _deadline_reminders__timedelta(days=days)).replace(hour=weekly.hour, minute=weekly.minute, second=0, microsecond=0)
        if weekly_target <= now:
            weekly_target += _deadline_reminders__timedelta(days=7)
        return min(reminder_target, weekly_target) if self.settings.enabled else weekly_target

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                now = self.now()
                delay = self._next_delay(now)
                if self._stop.wait(delay):
                    return
                self.scan()
                self.scan_weekly()
            except Exception:
                deadline_reminders.logger.exception('scheduler_error stage=loop')
                if self._stop.wait(60):
                    return

    def _next_delay(self, now: _deadline_reminders__datetime) -> float:
        """Poll safely; task eligibility is evaluated in each owner's timezone."""
        scheduled_delay = max(0.0, (self._next_run(now) - now).total_seconds())
        if self.settings.enabled and self.settings.interval_enabled:
            return min(float(self.settings.scan_interval_seconds), scheduled_delay)
        if self.settings.enabled:
            return min(float(self.settings.scan_interval_seconds), scheduled_delay)
        return scheduled_delay

    def scan(self, reminder_date: _deadline_reminders__date | None=None) -> dict[str, int]:
        if not self.settings.enabled:
            return {'scanned': 0, 'sent': 0, 'skipped': 0}
        now = self.now()
        scan_date = reminder_date.isoformat() if reminder_date else 'owner_local'
        deadline_reminders.logger.info('reminder_scan_started reminder_date=%s', scan_date)
        deadline_reminders.logger.info('follow_up_scan_started reminder_date=%s', scan_date)
        try:
            inspected = list(self.load_tasks())
            pending = [task for task in inspected if not task.completed and str(task.status or '').casefold() not in {'cancelled', 'canceled', 'closed'}]
            for task in inspected:
                if task.completed:
                    deadline_reminders.logger.info('follow_up_skipped_completed task_id=%s', task.task_id)
            deadline_reminders.logger.info('reminder_scan_tasks_inspected count=%d', len(inspected))
            batches: dict[tuple[str, str], list[tuple[deadline_reminders.ReminderTask, str, _deadline_reminders__date]]] = {}
            skipped = 0
            for task in pending:
                try:
                    local_now = now.astimezone(_deadline_reminders__ZoneInfo(task.timezone))
                except Exception:
                    local_now = now.astimezone(_deadline_reminders__ZoneInfo(self.settings.timezone))
                local_today = reminder_date or local_now.date()
                if reminder_date is None and (not self.settings.interval_enabled):
                    local_minutes = local_now.hour * 60 + local_now.minute
                    scheduled_minutes = self.settings.hour * 60 + self.settings.minute
                    if local_minutes < scheduled_minutes:
                        continue
                delta = (task.due_date - local_today).days
                category = None
                if delta < 0 and str(task.priority or '').casefold() == 'p1' and self.settings.overdue:
                    category = 'high_priority_overdue'
                elif delta <= -self.settings.escalation_days and self.settings.overdue:
                    category = 'escalated'
                elif delta < 0 and self.settings.overdue:
                    category = 'overdue'
                elif delta == 0 and self.settings.due_today:
                    category = 'today'
                elif delta == 1 and self.settings.due_tomorrow:
                    category = 'tomorrow'
                elif delta > 0 and delta * 24 <= self.settings.due_soon_hours:
                    category = 'due_soon'
                elif 2 <= delta <= self.settings.high_priority_days and str(task.priority or '').casefold() == 'p1':
                    category = 'high_priority'
                if not category:
                    continue
                if not self.store.claim(task, category, now, self.settings.repeat_hours):
                    skipped += 1
                    deadline_reminders.logger.info('follow_up_skipped_already_sent task_id=%s reminder_type=%s', task.task_id, category)
                    continue
                deadline_reminders.logger.info('follow_up_task_selected task_id=%s reminder_type=%s recipient=%s', task.task_id, category, 'configured_channel' if task.owner_id.startswith('channel:') else 'task_owner')
                batch_category = category if category in {'overdue', 'high_priority_overdue', 'escalated', 'high_priority', 'due_soon'} else 'deadline'
                batches.setdefault((task.owner_id, batch_category), []).append((task, category, local_today))
            sent = 0
            eligible = sum((len(entries) for entries in batches.values()))
            deadline_reminders.logger.info('reminder_scan_eligible_tasks count=%d', eligible)
            escalations = 0
            for (recipient_id, category), entries in batches.items():
                tasks = [task for task, _, _ in entries]
                display_today = entries[0][2]
                message_category = entries[0][1] if len(entries) == 1 else category
                try:
                    self.send(recipient_id, deadline_reminders.format_reminder(tasks, message_category, display_today))
                except Exception:
                    for task, condition, _ in entries:
                        self.store.release(task, condition)
                    deadline_reminders.logger.exception('follow_up_failed stage=notification category=%s', category)
                    continue
                for task, condition, _ in entries:
                    self.store.delivered(task, condition, now)
                    sent += 1
                    escalations += int(condition == 'escalated')
                    deadline_reminders.logger.info('follow_up_sent task_id=%s reminder_type=%s recipient=%s', task.task_id, condition, 'configured_channel' if recipient_id.startswith('channel:') else 'task_owner')
            deadline_reminders.logger.info('reminders_sent count=%d', sent)
            deadline_reminders.logger.info('reminders_skipped_already_sent count=%d', skipped)
            deadline_reminders.logger.info('reminder_escalations count=%d', escalations)
            return {'scanned': len(pending), 'sent': sent, 'skipped': skipped}
        except Exception:
            deadline_reminders.logger.exception('scheduler_error stage=scan reminder_date=%s', scan_date)
            raise

    def scan_weekly(self, now: _deadline_reminders__datetime | None=None) -> dict[str, int]:
        """Send the configured weekly report once for its Monday-Sunday period."""
        now = now or self.now()
        settings = self.weekly_settings
        if not settings.enabled:
            return {'sent': 0, 'skipped': 0}
        if not settings.channel or not settings.actor_id or (not self.build_weekly) or (not self.send_weekly):
            deadline_reminders.logger.error('scheduler_error stage=weekly_config reason=missing_channel_actor_or_callback')
            return {'sent': 0, 'skipped': 0}
        week_start = now.date() - _deadline_reminders__timedelta(days=now.weekday())
        scheduled = _deadline_reminders__datetime.combine(week_start + _deadline_reminders__timedelta(days=settings.day), _deadline_reminders__datetime.min.time(), now.tzinfo).replace(hour=settings.hour, minute=settings.minute)
        if now < scheduled:
            return {'sent': 0, 'skipped': 0}
        period_start = week_start
        period_end = period_start + _deadline_reminders__timedelta(days=6)
        try:
            delivery = self.build_weekly(period_start, period_end)
            if delivery is None:
                return {'sent': 0, 'skipped': 0}
            if not self.store.claim_weekly(delivery, now):
                deadline_reminders.logger.info('weekly_summary_skipped_duplicate period_start=%s channel_id=%s', period_start.isoformat(), delivery.channel_id)
                return {'sent': 0, 'skipped': 1}
            try:
                self.send_weekly(delivery.channel_id, delivery.message)
            except Exception:
                self.store.release_weekly(delivery)
                deadline_reminders.logger.exception('scheduler_error stage=weekly_notification channel_id=%s', delivery.channel_id)
                return {'sent': 0, 'skipped': 0}
            self.store.weekly_delivered(delivery, now)
            deadline_reminders.logger.info('weekly_summary_sent period_start=%s period_end=%s channel_id=%s', period_start.isoformat(), period_end.isoformat(), delivery.channel_id)
            return {'sent': 1, 'skipped': 0}
        except Exception:
            deadline_reminders.logger.exception('scheduler_error stage=weekly_summary')
            return {'sent': 0, 'skipped': 0}
deadline_reminders.DeadlineReminderScheduler = _deadline_reminders__DeadlineReminderScheduler


# visual_analytics.py
'Presentation-only visual analytics over authorized normalized tasks.'
from collections import Counter as _visual_analytics__Counter
visual_analytics.Counter = _visual_analytics__Counter
import calendar as _visual_analytics__month_calendar
visual_analytics.month_calendar = _visual_analytics__month_calendar
from dataclasses import dataclass as _visual_analytics__dataclass
visual_analytics.dataclass = _visual_analytics__dataclass
from datetime import date as _visual_analytics__date, timedelta as _visual_analytics__timedelta
visual_analytics.date = _visual_analytics__date
visual_analytics.timedelta = _visual_analytics__timedelta
from html import escape as _visual_analytics__escape
visual_analytics.escape = _visual_analytics__escape
from io import BytesIO as _visual_analytics__BytesIO
visual_analytics.BytesIO = _visual_analytics__BytesIO
_visual_analytics__plt = None
_visual_analytics__Rectangle = None
_visual_analytics__FancyBboxPatch = None


def _visual_analytics__load_matplotlib():
    """Load the optional PNG renderer only when a visual is requested."""
    global _visual_analytics__plt, _visual_analytics__Rectangle, _visual_analytics__FancyBboxPatch
    if _visual_analytics__plt is None:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        from matplotlib.patches import FancyBboxPatch, Rectangle
        _visual_analytics__plt = pyplot
        _visual_analytics__Rectangle = Rectangle
        _visual_analytics__FancyBboxPatch = FancyBboxPatch
        visual_analytics.matplotlib = matplotlib
        visual_analytics.plt = pyplot
        visual_analytics.FancyBboxPatch = FancyBboxPatch
        visual_analytics.Rectangle = Rectangle
_visual_analytics__MODES = {'text', 'chart', 'dashboard', 'table'}
visual_analytics.MODES = _visual_analytics__MODES
_visual_analytics__KINDS = {'workload', 'priority', 'completion', 'deadlines', 'completed_trend', 'created_trend', 'all_tasks', 'overdue_tasks', 'upcoming_tasks', 'dashboard'}
visual_analytics.KINDS = _visual_analytics__KINDS
@_visual_analytics__dataclass(frozen=True)
class _visual_analytics__VisualDataset:
    kind: str
    title: str
    scope: str
    series: tuple[tuple[str, int], ...]
    record_count: int
    summary: str

    @property
    def meaningful(self):
        return bool(self.series) and sum((value for _, value in self.series)) > 0
visual_analytics.VisualDataset = _visual_analytics__VisualDataset
@_visual_analytics__dataclass(frozen=True)
class _visual_analytics__TableDataset:
    title: str
    scope: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    summary: str

    @property
    def meaningful(self):
        return bool(self.rows)
visual_analytics.TableDataset = _visual_analytics__TableDataset
def _visual_analytics__choose_response_mode(*, explicit_visual=False, dashboard=False, task_level=False, value_count=0):
    """Choose presentation mode without consulting an LLM."""
    if task_level:
        return 'table'
    if dashboard:
        return 'dashboard' if value_count else 'text'
    if explicit_visual and value_count:
        return 'chart'
    return 'chart' if value_count >= 2 else 'text'
visual_analytics.choose_response_mode = _visual_analytics__choose_response_mode
def _visual_analytics__build_dataset(tasks, kind, *, today: _visual_analytics__date, name_for_user) -> visual_analytics.VisualDataset:
    """Transform normalized tasks into factual chart data without business I/O."""
    tasks = list(tasks)
    pending = [task for task in tasks if not task.completed]
    if kind == 'priority':
        counts = _visual_analytics__Counter((task.priority or 'Unspecified' for task in pending))
        rows = tuple(((key, counts[key]) for key in ('P1', 'P2', 'P3', 'P4', 'Unspecified') if counts[key]))
        summary = f'{len(pending)} pending task{('s' if len(pending) != 1 else '')} are shown by priority.' if pending else 'No pending priority data is available to visualize.'
        return visual_analytics.VisualDataset(kind, 'Pending Task Priority Distribution', 'pending tasks', rows, len(pending), summary)
    if kind == 'completion':
        completed = len(tasks) - len(pending)
        rows = (('Pending', len(pending)), ('Completed', completed)) if tasks else ()
        rate = round(completed * 100 / len(tasks), 1) if tasks else 0
        summary = f'{completed} of {len(tasks)} tasks are completed ({rate:g}%).' if tasks else 'No task status data is available to visualize.'
        return visual_analytics.VisualDataset(kind, 'Current Task Completion', 'all authorized tasks', rows, len(tasks), summary)
    if kind == 'deadlines':
        counts = _visual_analytics__Counter()
        for task in pending:
            due = task.due_date
            if not due:
                counts['No due date'] += 1
            elif due < today:
                counts['Overdue'] += 1
            elif due == today:
                counts['Due today'] += 1
            elif due == today + _visual_analytics__timedelta(days=1):
                counts['Next 24h'] += 1
            elif due == today + _visual_analytics__timedelta(days=2):
                counts['Next 48h'] += 1
            elif due <= today + _visual_analytics__timedelta(days=6):
                counts['This week'] += 1
            else:
                counts['Later'] += 1
        order = ('Overdue', 'Due today', 'Next 24h', 'Next 48h', 'This week', 'Later', 'No due date')
        rows = tuple(((key, counts[key]) for key in order if counts[key]))
        summary = f'{len(pending)} pending task{('s' if len(pending) != 1 else '')} are shown by deadline window.' if pending else 'No pending deadline data is available to visualize.'
        return visual_analytics.VisualDataset(kind, 'Pending Task Deadline Distribution', 'pending tasks', rows, len(pending), summary)
    raise ValueError(f'Unsupported visual analytics kind: {kind}')
visual_analytics.build_dataset = _visual_analytics__build_dataset
def _visual_analytics__workload_dataset(report) -> visual_analytics.VisualDataset:
    """Adapt the existing workload intelligence report without recalculating it."""
    rows = tuple(sorted(((row['name'], row['pending']) for row in report.rows.values() if row['pending']), key=lambda value: (-value[1], value[0].casefold())))
    total = sum((value for _, value in rows))
    leader = rows[0] if rows else None
    summary = f'{leader[0]} currently has the largest pending workload with {leader[1]} task{('s' if leader[1] != 1 else '')}.' if leader else 'No pending workload is available to visualize.'
    return visual_analytics.VisualDataset('workload', 'Pending Workload by Owner', 'pending tasks', rows, total, summary)
visual_analytics.workload_dataset = _visual_analytics__workload_dataset
def _visual_analytics__render_deadline_heatmap_png(points, *, today) -> bytes:
    """Render a restrained visual heatmap from precomputed deadline facts."""
    _visual_analytics__load_matplotlib()
    values = list(points)
    fig, ax = _visual_analytics__plt.subplots(figsize=(14, 5.5), facecolor='#F4F7FB')
    ax.set_facecolor('#F4F7FB')
    if values:
        labels = [day.strftime('%d %b') for day, _, _ in values]
        totals = [count for _, count, _ in values]
        p1 = [count for _, _, count in values]
        colors = ['#A63D40' if high else '#365F91' for high in p1]
        bars = ax.bar(labels, totals, color=colors, width=0.68)
        for bar, total, high in zip(bars, totals, p1):
            ax.text(bar.get_x() + bar.get_width() / 2, total + 0.08, f'{total} task{('s' if total != 1 else '')} · {high} P1', ha='center', va='bottom', fontsize=9, color='#24344D')
        ax.set_ylim(0, max(totals) + 1.4)
    else:
        ax.text(0.5, 0.5, 'No upcoming authorized deadlines', ha='center', va='center', transform=ax.transAxes, fontsize=16, color='#52637A')
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(f'DEADLINE HEATMAP  |  FROM {today.strftime('%d %b %Y').upper()}', loc='left', fontsize=18, fontweight='bold', color='#17263C', pad=20)
    ax.set_ylabel('Deadline count', color='#52637A')
    ax.grid(axis='y', color='#DCE3EC', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.tight_layout(pad=2)
    return visual_analytics._finish_png(fig)
visual_analytics.render_deadline_heatmap_png = _visual_analytics__render_deadline_heatmap_png
def _visual_analytics__render_control_tower_png(tower, *, name_for_user, show_owner=True, show_priority=True, show_due=True) -> bytes:
    """Render the aggregated Control Tower without recalculating intelligence."""
    _visual_analytics__load_matplotlib()
    fig = _visual_analytics__plt.figure(figsize=(18, 11), facecolor='#F3F6FA')
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 100)
    canvas.set_ylim(0, 100)
    canvas.axis('off')
    canvas.text(4, 95, 'ACTION ITEM CONTROL TOWER', fontsize=24, fontweight='bold', color='#17263C', va='center')
    canvas.text(4, 91.5, 'Executive operational view · authorized Action Items', fontsize=10.5, color='#617188')
    summary = tower.summary
    metrics = [('PENDING', summary.pending), ('CRITICAL', sum((risk.level == 'Critical' for risk in tower.risks)))]
    if show_due:
        metrics.extend((('OVERDUE', summary.overdue), ('DUE TODAY', summary.due_today)))
    if show_priority:
        metrics.append(('P1', summary.priority_counts.get('P1', 0)))
    if show_owner:
        metrics.append(('UNASSIGNED', summary.unassigned))
    card_width = 91 / max(1, len(metrics))
    for index, (label, value) in enumerate(metrics):
        x = 4 + index * card_width
        canvas.add_patch(_visual_analytics__FancyBboxPatch((x, 82), card_width - 1.2, 7, boxstyle='round,pad=0.3,rounding_size=.7', facecolor='#FFFFFF', edgecolor='#D7E0EB', linewidth=1))
        canvas.text(x + 1.5, 87, label, fontsize=8.5, fontweight='bold', color='#66768C')
        canvas.text(x + 1.5, 83.4, str(value), fontsize=21, fontweight='bold', color='#203550')

    def panel(x, y, width, height, title):
        canvas.add_patch(_visual_analytics__FancyBboxPatch((x, y), width, height, boxstyle='round,pad=.5,rounding_size=.8', facecolor='#FFFFFF', edgecolor='#D7E0EB', linewidth=1))
        canvas.text(x + 1.5, y + height - 2.4, title, fontsize=11, fontweight='bold', color='#203550', va='center')
    panel(4, 48, 57, 31, 'RISK RADAR')
    visible_risks = [risk for risk in tower.risks if risk.level != 'Low'][:6]
    if visible_risks:
        for index, risk in enumerate(visible_risks):
            y = 73.5 - index * 4.2
            risk_color = {'Critical': '#9D3940', 'High': '#B66A32', 'Medium': '#9A7B2F'}.get(risk.level, '#52637A')
            canvas.add_patch(_visual_analytics__Rectangle((5.5, y - 1.8), 0.45, 2.8, color=risk_color))
            canvas.text(6.7, y, visual_analytics._short(risk.task.name, 34), fontsize=9.5, fontweight='bold', color='#263A54', va='center')
            meta = []
            if show_owner:
                meta.append(', '.join((name_for_user(value) for value in risk.task.owner_ids)) or 'Unassigned')
            if show_priority:
                meta.append(risk.task.priority or 'No priority')
            if show_due and risk.task.due_date:
                meta.append(risk.task.due_date.strftime('%d %b'))
            canvas.text(31, y, ' · '.join(meta), fontsize=8.5, color='#66768C', va='center')
            canvas.text(58.5, y, risk.level.upper(), fontsize=8.5, fontweight='bold', color=risk_color, ha='right', va='center')
    else:
        canvas.text(6, 69, 'No material risk signal is visible.', fontsize=10, color='#66768C')
    panel(63, 48, 33, 31, 'WORKLOAD PRESSURE')
    for index, row in enumerate(summary.workload[:6]):
        y = 73.5 - index * 4.2
        name = name_for_user(row.owner_id) if row.owner_id and show_owner else 'Unassigned' if show_owner else 'Restricted'
        pressure = tower.workload_labels.get(row.owner_id, 'Medium')
        canvas.text(65, y, visual_analytics._short(name, 18), fontsize=9.2, fontweight='bold', color='#263A54', va='center')
        canvas.text(78, y, f'{row.pending} pending', fontsize=8.5, color='#66768C', va='center')
        canvas.text(94, y, pressure.upper(), fontsize=8.2, fontweight='bold', color='#365F91', ha='right', va='center')
    panel(4, 19, 45, 26, 'BOTTLENECKS')
    bottlenecks = tower.bottlenecks[:4]
    for index, item in enumerate(bottlenecks):
        y = 39 - index * 5.2
        canvas.text(6, y, f'{index + 1}. {visual_analytics._short(item.title, 32)}', fontsize=9.3, fontweight='bold', color='#263A54')
        canvas.text(8, y - 2, visual_analytics._short(item.detail, 62), fontsize=8.2, color='#66768C')
    if not bottlenecks:
        canvas.text(6, 35, 'No evidence-backed bottlenecks identified.', fontsize=9.5, color='#66768C')
    panel(51, 19, 45, 26, 'RECOMMENDED ACTIONS')
    for index, value in enumerate(tower.recommendations[:4]):
        y = 39 - index * 5.2
        canvas.text(53, y, str(index + 1), fontsize=9, fontweight='bold', color='#365F91')
        canvas.text(55.5, y, visual_analytics._short(value, 58), fontsize=9, color='#263A54')
    if not tower.recommendations:
        canvas.text(53, 35, 'No immediate recommendation is supported.', fontsize=9.5, color='#66768C')
    canvas.text(4, 13.5, 'EXECUTIVE SUMMARY', fontsize=10.5, fontweight='bold', color='#203550')
    executive = f'{summary.pending} pending'
    if show_due:
        executive += f' · {summary.overdue} overdue · {summary.due_within_7d} due within 7 days'
    executive += f' · {len(visible_risks)} material risks · {len(tower.bottlenecks)} bottlenecks'
    canvas.text(4, 10.5, executive, fontsize=11, color='#40536C')
    if tower.unavailable:
        canvas.text(4, 6.5, 'Unavailable sections: ' + ', '.join(tower.unavailable), fontsize=8.5, color='#8A5A3B')
    canvas.text(96, 4, 'READ-ONLY', ha='right', fontsize=8.5, fontweight='bold', color='#66768C')
    return visual_analytics._finish_png(fig)
visual_analytics.render_control_tower_png = _visual_analytics__render_control_tower_png
def _visual_analytics___panel(title, rows, x, y, width, *, color='#4C78A8'):
    height = max(180, 90 + len(rows) * 48)
    maximum = max((value for _, value in rows), default=1)
    parts = [f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="#FFFFFF"/>', f'<text x="{x + 28}" y="{y + 42}" class="panel">{_visual_analytics__escape(title)}</text>']
    label_width = min(260, int(width * 0.38))
    bar_width = max(120, width - label_width - 105)
    for index, (label, value) in enumerate(rows):
        row_y = y + 82 + index * 48
        scaled = 0 if maximum <= 0 else max(3, int(bar_width * value / maximum))
        clipped = label if len(label) <= 26 else label[:25] + '…'
        parts.extend((f'<text x="{x + 28}" y="{row_y + 17}" class="label">{_visual_analytics__escape(clipped)}</text>', f'<rect x="{x + label_width}" y="{row_y}" width="{bar_width}" height="22" rx="6" fill="#E8EDF5"/>', f'<rect x="{x + label_width}" y="{row_y}" width="{scaled}" height="22" rx="6" fill="{color}"/>', f'<text x="{x + label_width + bar_width + 16}" y="{row_y + 17}" class="value">{value}</text>'))
    return (''.join(parts), height)
visual_analytics._panel = _visual_analytics___panel
def _visual_analytics__render_svg(dataset: visual_analytics.VisualDataset) -> str:
    """Render an exact, accessible vector bar chart with no temporary file."""
    width = 1200
    panel, panel_height = visual_analytics._panel(dataset.title, dataset.series, 55, 105, 1090)
    height = panel_height + 190
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_visual_analytics__escape(dataset.title)}"><style>.title{{font:700 34px Arial,sans-serif;fill:#172B4D}}.scope{{font:18px Arial,sans-serif;fill:#5E6C84}}.panel{{font:700 23px Arial,sans-serif;fill:#172B4D}}.label{{font:18px Arial,sans-serif;fill:#344563}}.value{{font:700 18px Arial,sans-serif;fill:#172B4D}}</style><rect width="100%" height="100%" fill="#F4F6F8"/><text x="55" y="52" class="title">{_visual_analytics__escape(dataset.title)}</text><text x="55" y="80" class="scope">Scope: {_visual_analytics__escape(dataset.scope)}</text>{panel}</svg>'
visual_analytics.render_svg = _visual_analytics__render_svg
def _visual_analytics__render_donut_svg(dataset: visual_analytics.VisualDataset) -> str:
    """Render a small-category part-to-whole chart with exact values."""
    total = sum((value for _, value in dataset.series))
    if total <= 0:
        raise ValueError('A donut chart requires positive data.')
    colors = ('#E45756', '#4C78A8', '#72B7B2', '#F2CF5B', '#B279A2')
    radius, circumference = (145, 2 * 3.141592653589793 * 145)
    offset, circles, legend = (0.0, [], [])
    for index, (label, value) in enumerate(dataset.series):
        length = circumference * value / total
        color = colors[index % len(colors)]
        circles.append(f'<circle cx="300" cy="300" r="{radius}" fill="none" stroke="{color}" stroke-width="70" stroke-dasharray="{length:.3f} {circumference - length:.3f}" stroke-dashoffset="{-offset:.3f}" transform="rotate(-90 300 300)"/>')
        legend.append(f'<rect x="610" y="{185 + index * 62}" width="24" height="24" rx="5" fill="{color}"/><text x="650" y="{204 + index * 62}" class="label">{_visual_analytics__escape(label)}: {value}</text>')
        offset += length
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" viewBox="0 0 1200 650" role="img"><style>.title{{font:700 34px Arial,sans-serif;fill:#172B4D}}.scope{{font:18px Arial,sans-serif;fill:#5E6C84}}.label{{font:21px Arial,sans-serif;fill:#344563}}.total{{font:700 42px Arial,sans-serif;fill:#172B4D}}.small{{font:18px Arial,sans-serif;fill:#5E6C84}}</style><rect width="100%" height="100%" fill="#F4F6F8"/><text x="55" y="58" class="title">{_visual_analytics__escape(dataset.title)}</text><text x="55" y="88" class="scope">Scope: {_visual_analytics__escape(dataset.scope)}</text><rect x="55" y="120" width="1090" height="475" rx="18" fill="#FFFFFF"/>' + ''.join(circles) + f'<text x="300" y="295" text-anchor="middle" class="total">{total}</text><text x="300" y="328" text-anchor="middle" class="small">tasks</text>' + ''.join(legend) + '</svg>'
visual_analytics.render_donut_svg = _visual_analytics__render_donut_svg
def _visual_analytics__render_line_svg(dataset: visual_analytics.VisualDataset) -> str:
    """Render a line only from actual ordered time-series observations."""
    rows = list(dataset.series)
    if len(rows) < 2:
        raise ValueError('A line chart requires at least two observed periods.')
    width, height = (1200, 650)
    x0, y0, plot_width, plot_height = (120, 520, 980, 350)
    maximum = max((value for _, value in rows)) or 1
    points = []
    labels = []
    for index, (label, value) in enumerate(rows):
        x = x0 + plot_width * index / (len(rows) - 1)
        y = y0 - plot_height * value / maximum
        points.append(f'{x:.1f},{y:.1f}')
        labels.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#4C78A8"/><text x="{x:.1f}" y="{y - 15:.1f}" text-anchor="middle" class="value">{value}</text><text x="{x:.1f}" y="{y0 + 34}" text-anchor="middle" class="axis">{_visual_analytics__escape(label)}</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><style>.title{{font:700 34px Arial,sans-serif;fill:#172B4D}}.scope{{font:18px Arial,sans-serif;fill:#5E6C84}}.axis{{font:15px Arial,sans-serif;fill:#5E6C84}}.value{{font:700 17px Arial,sans-serif;fill:#172B4D}}</style><rect width="100%" height="100%" fill="#F4F6F8"/><text x="55" y="58" class="title">{_visual_analytics__escape(dataset.title)}</text><text x="55" y="88" class="scope">Scope: {_visual_analytics__escape(dataset.scope)}</text><rect x="55" y="120" width="1090" height="470" rx="18" fill="#FFFFFF"/><line x1="{x0}" y1="{y0}" x2="{x0 + plot_width}" y2="{y0}" stroke="#B3BAC5" stroke-width="2"/><polyline points="{' '.join(points)}" fill="none" stroke="#4C78A8" stroke-width="7" stroke-linejoin="round" stroke-linecap="round"/>' + ''.join(labels) + '</svg>'
visual_analytics.render_line_svg = _visual_analytics__render_line_svg
def _visual_analytics__render_chart(dataset: visual_analytics.VisualDataset) -> str:
    if dataset.kind in {'priority', 'completion'}:
        return visual_analytics.render_donut_svg(dataset)
    if dataset.kind in {'completed_trend', 'created_trend'}:
        return visual_analytics.render_line_svg(dataset)
    return visual_analytics.render_svg(dataset)
visual_analytics.render_chart = _visual_analytics__render_chart
_visual_analytics__COLORS = ('#4C78A8', '#E45756', '#72B7B2', '#F2CF5B', '#B279A2', '#FF9DA6')
visual_analytics.COLORS = _visual_analytics__COLORS
def _visual_analytics___finish_png(fig):
    output = _visual_analytics__BytesIO()
    fig.savefig(output, format='png', dpi=180, bbox_inches='tight', facecolor='#F7F9FC')
    _visual_analytics__plt.close(fig)
    return output.getvalue()
visual_analytics._finish_png = _visual_analytics___finish_png
def _visual_analytics___style_axis(ax, title, scope):
    ax.set_title(title, loc='left', fontsize=18, fontweight='bold', color='#172B4D', pad=22)
    ax.text(0, 1.02, f'Scope: {scope}', transform=ax.transAxes, fontsize=10, color='#5E6C84', va='bottom')
    ax.set_facecolor('#FFFFFF')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color('#C1C7D0')
    ax.spines['bottom'].set_color('#C1C7D0')
    ax.tick_params(colors='#344563', labelsize=10)
visual_analytics._style_axis = _visual_analytics___style_axis
_visual_analytics___CALENDAR_PRIORITY = {'P1': ('#8C2F39', '#F9E9EB'), 'P2': ('#9A5B13', '#FFF3E3'), 'P3': ('#315D75', '#EAF3F7'), 'P4': ('#486B56', '#EDF5F0')}
visual_analytics._CALENDAR_PRIORITY = _visual_analytics___CALENDAR_PRIORITY
def _visual_analytics___short(value, limit):
    value = str(value or '')
    return value if len(value) <= limit else value[:limit - 1].rstrip() + '…'
visual_analytics._short = _visual_analytics___short
def _visual_analytics__render_team_calendar_png(tasks, clocks, *, today, name_for_user, show_owner=True, show_priority=True) -> bytes:
    """Render an enterprise month grid from an already-authorized task snapshot."""
    _visual_analytics__load_matplotlib()
    tasks = [task for task in tasks if not task.completed and task.due_date]
    month_tasks = [task for task in tasks if (task.due_date.year, task.due_date.month) == (today.year, today.month)]
    overdue = [task for task in tasks if task.due_date < today]
    due_today = [task for task in tasks if task.due_date == today]
    due_week = [task for task in tasks if today <= task.due_date <= today + _visual_analytics__timedelta(days=6)]
    grouped = {}
    for task in month_tasks:
        grouped.setdefault(task.due_date.day, []).append(task)
    collisions = sum((len(values) >= 3 for values in grouped.values()))
    pressure = _visual_analytics__Counter()
    for task in tasks:
        if show_owner:
            for owner in task.owner_ids or ('Unassigned',):
                pressure[owner] += 3 if show_priority and task.priority == 'P1' else 2 if show_priority and task.priority == 'P2' else 1
    fig = _visual_analytics__plt.figure(figsize=(18, 11), facecolor='#F4F6F8')
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')
    ax.text(0.55, 10.48, 'TEAM CALENDAR', fontsize=25, fontweight='bold', color='#172B4D')
    ax.text(17.45, 10.48, today.strftime('%B %Y').upper(), fontsize=17, fontweight='bold', color='#344563', ha='right')
    ax.text(0.55, 10.08, 'Authorized deadlines and team availability', fontsize=10.5, color='#5E6C84')
    left, bottom, width, height = (0.55, 0.62, 12.7, 9.05)
    header_h = 0.52
    weeks = _visual_analytics__month_calendar.Calendar(firstweekday=0).monthdayscalendar(today.year, today.month)
    cell_w, cell_h = (width / 7, (height - header_h) / len(weeks))
    for index, label in enumerate(('MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN')):
        x = left + index * cell_w
        ax.add_patch(_visual_analytics__Rectangle((x, bottom + height - header_h), cell_w, header_h, facecolor='#E9EDF3', edgecolor='#D3D9E2', linewidth=1))
        ax.text(x + 0.12, bottom + height - 0.34, label, fontsize=9, fontweight='bold', color='#44546A')
    for row_index, week in enumerate(weeks):
        y = bottom + height - header_h - (row_index + 1) * cell_h
        for column, day in enumerate(week):
            x = left + column * cell_w
            background = '#FBFCFE' if column < 5 else '#F7F8FA'
            ax.add_patch(_visual_analytics__Rectangle((x, y), cell_w, cell_h, facecolor=background, edgecolor='#D9DEE7', linewidth=1))
            if not day:
                continue
            is_today = day == today.day
            if is_today:
                ax.add_patch(_visual_analytics__FancyBboxPatch((x + 0.08, y + cell_h - 0.39), 0.38, 0.28, boxstyle='round,pad=.02,rounding_size=.05', facecolor='#274C77', edgecolor='none'))
            ax.text(x + 0.13, y + cell_h - 0.26, str(day), fontsize=9.5, fontweight='bold', color='#FFFFFF' if is_today else '#344563')
            values = sorted(grouped.get(day, []), key=lambda task: ({'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4}.get(task.priority, 9), task.name.casefold()))
            card_height = min(0.42, max(0.31, (cell_h - 0.5) / 3.2))
            for card_index, task in enumerate(values[:3]):
                card_y = y + cell_h - 0.52 - (card_index + 1) * card_height
                foreground, background = visual_analytics._CALENDAR_PRIORITY.get(task.priority, ('#44546A', '#EEF1F5')) if show_priority else ('#44546A', '#EEF1F5')
                if task.due_date < today:
                    foreground, background = ('#8C2F39', '#F7E5E7')
                ax.add_patch(_visual_analytics__FancyBboxPatch((x + 0.09, card_y), cell_w - 0.18, card_height - 0.045, boxstyle='round,pad=.025,rounding_size=.05', facecolor=background, edgecolor=foreground, linewidth=0.8))
                owner = ', '.join((name_for_user(value) for value in task.owner_ids)) or 'Unassigned' if show_owner else 'Owner restricted'
                status = 'Overdue' if task.due_date < today else 'Due today' if task.due_date == today else 'Pending'
                ax.text(x + 0.16, card_y + card_height - 0.16, visual_analytics._short(task.name, 23), fontsize=7.3, fontweight='bold', color='#172B4D', va='top')
                priority = task.priority or '—' if show_priority else 'Priority restricted'
                ax.text(x + 0.16, card_y + 0.08, visual_analytics._short(f'{owner}  {priority}  {status}', 29), fontsize=6.3, color=foreground, va='bottom')
            if len(values) > 3:
                ax.text(x + 0.13, y + 0.08, f'+{len(values) - 3} more', fontsize=6.5, color='#5E6C84')
    panel_x, panel_w = (13.6, 3.85)
    ax.add_patch(_visual_analytics__FancyBboxPatch((panel_x, 6.85), panel_w, 2.82, boxstyle='round,pad=.08,rounding_size=.12', facecolor='#FFFFFF', edgecolor='#D9DEE7'))
    ax.text(panel_x + 0.25, 9.28, 'CALENDAR SUMMARY', fontsize=11, fontweight='bold', color='#172B4D')
    metrics = (('Active deadlines', len(tasks)), ('Overdue', len(overdue)), ('Due today', len(due_today)), ('Due this week', len(due_week)), ('High priority', sum((task.priority == 'P1' for task in tasks)) if show_priority else 'Restricted'), ('Date collisions', collisions))
    for index, (label, value) in enumerate(metrics):
        row_y = 8.85 - index * 0.37
        ax.text(panel_x + 0.25, row_y, label, fontsize=8.2, color='#5E6C84')
        ax.text(panel_x + panel_w - 0.25, row_y, str(value), fontsize=9, fontweight='bold', color='#172B4D', ha='right')
    ax.add_patch(_visual_analytics__FancyBboxPatch((panel_x, 4.72), panel_w, 1.8, boxstyle='round,pad=.08,rounding_size=.12', facecolor='#FFFFFF', edgecolor='#D9DEE7'))
    ax.text(panel_x + 0.25, 6.14, 'DEADLINE PRESSURE', fontsize=11, fontweight='bold', color='#172B4D')
    for index, (owner, score) in enumerate(pressure.most_common(3)):
        label = 'Unassigned' if owner == 'Unassigned' else name_for_user(owner)
        row_y = 5.75 - index * 0.4
        ax.text(panel_x + 0.25, row_y, visual_analytics._short(label, 20), fontsize=8.2, color='#344563')
        ax.text(panel_x + panel_w - 0.25, row_y, str(score), fontsize=8.5, fontweight='bold', color='#172B4D', ha='right')
        ax.add_patch(_visual_analytics__Rectangle((panel_x + 0.25, row_y - 0.15), (panel_w - 0.5) * score / max(pressure.values(), default=1), 0.055, facecolor='#6B7C93', edgecolor='none'))
    ax.add_patch(_visual_analytics__FancyBboxPatch((panel_x, 0.62), panel_w, 3.78, boxstyle='round,pad=.08,rounding_size=.12', facecolor='#FFFFFF', edgecolor='#D9DEE7'))
    ax.text(panel_x + 0.25, 4.02, 'TEAM TIME ZONES', fontsize=11, fontweight='bold', color='#172B4D')
    for index, clock in enumerate(list(clocks)[:6]):
        row_y = 3.6 - index * 0.48
        ax.text(panel_x + 0.25, row_y, visual_analytics._short(clock.name, 18), fontsize=8.2, fontweight='bold', color='#344563')
        if clock.local_time:
            detail = f'{clock.local_time.strftime('%H:%M')}  {clock.utc_offset}'
            status = clock.availability
        else:
            detail, status = ('Time zone not configured', '')
        ax.text(panel_x + panel_w - 0.25, row_y, detail, fontsize=7.3, color='#5E6C84', ha='right')
        if status:
            ax.text(panel_x + 0.25, row_y - 0.18, visual_analytics._short(status, 34), fontsize=6.5, color='#6B778C')
    ax.text(17.45, 0.28, 'Read-only view · No task changes were made', fontsize=7.3, color='#6B778C', ha='right')
    return visual_analytics._finish_png(fig)
visual_analytics.render_team_calendar_png = _visual_analytics__render_team_calendar_png
def _visual_analytics__render_team_clock_png(clocks, *, requester_clock=None) -> bytes:
    """Render configured team clocks and availability as a compact visual panel."""
    _visual_analytics__load_matplotlib()
    clocks = list(clocks)
    height = max(4.5, 2.25 + 0.72 * len(clocks))
    fig, ax = _visual_analytics__plt.subplots(figsize=(13, height), facecolor='#F4F6F8')
    ax.axis('off')
    ax.set_xlim(0, 13)
    ax.set_ylim(0, height)
    ax.text(0.45, height - 0.45, 'TEAM TIME ZONES', fontsize=21, fontweight='bold', color='#172B4D')
    ax.text(0.45, height - 0.8, 'Configured global working context', fontsize=10, color='#5E6C84')
    headers = ((0.55, 'PERSON'), (3.3, 'LOCATION / ZONE'), (7.35, 'LOCAL TIME'), (9.65, 'UTC OFFSET'), (11.1, 'WORK STATUS'))
    header_y = height - 1.35
    ax.add_patch(_visual_analytics__Rectangle((0.4, header_y - 0.22), 12.2, 0.48, facecolor='#E9EDF3', edgecolor='none'))
    for x, label in headers:
        ax.text(x, header_y, label, fontsize=8, fontweight='bold', color='#44546A', va='center')
    for index, clock in enumerate(clocks):
        y = header_y - 0.62 - index * 0.66
        ax.add_patch(_visual_analytics__Rectangle((0.4, y - 0.25), 12.2, 0.58, facecolor='#FFFFFF' if index % 2 == 0 else '#F8F9FB', edgecolor='#E1E5EB', linewidth=0.6))
        zone = clock.location or clock.timezone_name or 'Time zone not configured'
        local = clock.local_time.strftime('%a %d %b  %H:%M') if clock.local_time else 'Not configured'
        status = clock.availability if clock.local_time else 'Unavailable'
        values = ((0.55, clock.name, True), (3.3, zone, False), (7.35, local, False), (9.65, clock.utc_offset or '—', False), (11.1, status, False))
        for x, value, bold in values:
            ax.text(x, y, visual_analytics._short(value, 28 if x == 3.3 else 22), fontsize=8.2, fontweight='bold' if bold else 'normal', color='#172B4D' if bold else '#44546A', va='center')
    ax.text(12.55, 0.2, 'Time-zone data is never inferred', fontsize=7.5, color='#6B778C', ha='right')
    return visual_analytics._finish_png(fig)
visual_analytics.render_team_clock_png = _visual_analytics__render_team_clock_png
def _visual_analytics__render_chart_png(dataset: visual_analytics.VisualDataset, chart_type='auto') -> bytes:
    """Render a professional PNG from exact structured values."""
    if not dataset.meaningful:
        raise ValueError('No meaningful chart data is available.')
    _visual_analytics__load_matplotlib()
    chart_type = chart_type if chart_type != 'auto' else 'pie' if dataset.kind in {'priority', 'completion'} else 'line' if dataset.kind in {'completed_trend', 'created_trend'} else 'bar'
    labels = [label for label, _ in dataset.series]
    values = [value for _, value in dataset.series]
    if chart_type == 'line' and len(values) < 2:
        raise ValueError('A line chart requires at least two real observations.')
    if chart_type == 'pie':
        fig, ax = _visual_analytics__plt.subplots(figsize=(8.5, 6.2), layout='constrained')
        wedges, _, autotexts = ax.pie(values, startangle=90, colors=visual_analytics.COLORS[:len(values)], autopct=lambda percent: f'{percent:.1f}%' if percent >= 3 else '', wedgeprops={'width': 0.55, 'edgecolor': 'white', 'linewidth': 2}, textprops={'color': '#172B4D', 'fontsize': 10})
        ax.legend(wedges, [f'{label} · {value}' for label, value in dataset.series], loc='center left', bbox_to_anchor=(0.88, 0.5), frameon=False, fontsize=10)
        ax.set_title(dataset.title, loc='left', fontsize=18, fontweight='bold', color='#172B4D', pad=20)
        ax.text(0, 1.01, f'Scope: {dataset.scope}', transform=ax.transAxes, fontsize=10, color='#5E6C84')
        for value in autotexts:
            value.set_fontweight('bold')
        return visual_analytics._finish_png(fig)
    if chart_type == 'line':
        fig, ax = _visual_analytics__plt.subplots(figsize=(10, 5.8), layout='constrained')
        x = list(range(len(values)))
        ax.plot(x, values, color=visual_analytics.COLORS[0], linewidth=3, marker='o', markersize=7)
        ax.set_xticks(x, labels, rotation=30, ha='right')
        ax.set_ylabel('Task count', color='#344563')
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', color='#DFE1E6', linewidth=0.8)
        for index, value in enumerate(values):
            ax.annotate(str(value), (index, value), xytext=(0, 9), textcoords='offset points', ha='center', fontweight='bold')
        visual_analytics._style_axis(ax, dataset.title, dataset.scope)
        return visual_analytics._finish_png(fig)
    if chart_type == 'table':
        table = visual_analytics.TableDataset(dataset.title, dataset.scope, ('Category', 'Count'), tuple(((label, str(value)) for label, value in dataset.series)), dataset.summary)
        return visual_analytics.render_table_png(table)
    fig, ax = _visual_analytics__plt.subplots(figsize=(10, 5.8), layout='constrained')
    bars = ax.bar(labels, values, color=visual_analytics.COLORS[:len(values)], width=0.65)
    ax.set_xlabel('Owner' if dataset.kind == 'workload' else 'Category', color='#344563')
    ax.set_ylabel('Pending task count' if dataset.kind == 'workload' else 'Task count', color='#344563')
    ax.set_ylim(bottom=0, top=max(values) * 1.22 if max(values) else 1)
    ax.grid(axis='y', color='#DFE1E6', linewidth=0.8)
    ax.bar_label(bars, labels=[str(value) for value in values], padding=4, fontsize=10, fontweight='bold', color='#172B4D')
    ax.tick_params(axis='x', rotation=20)
    visual_analytics._style_axis(ax, dataset.title, dataset.scope)
    return visual_analytics._finish_png(fig)
visual_analytics.render_chart_png = _visual_analytics__render_chart_png
def _visual_analytics__render_table_png(dataset: visual_analytics.TableDataset) -> bytes:
    if not dataset.meaningful:
        raise ValueError('No table rows are available.')
    _visual_analytics__load_matplotlib()
    height = max(3.8, 1.8 + 0.48 * len(dataset.rows))
    fig, ax = _visual_analytics__plt.subplots(figsize=(12, height), layout='constrained')
    ax.axis('off')
    ax.set_title(dataset.title, loc='left', fontsize=18, fontweight='bold', color='#172B4D', pad=20)
    ax.text(0, 0.98, f'Scope: {dataset.scope}', transform=ax.transAxes, fontsize=10, color='#5E6C84', va='top')
    table = ax.table(cellText=dataset.rows, colLabels=dataset.columns, loc='center', cellLoc='left', colLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor('#DFE1E6')
        cell.set_facecolor('#E9F2FF' if row == 0 else '#FFFFFF')
        if row == 0:
            cell.set_text_props(weight='bold', color='#172B4D')
    return visual_analytics._finish_png(fig)
visual_analytics.render_table_png = _visual_analytics__render_table_png
def _visual_analytics__render_dashboard_png(datasets) -> bytes:
    datasets = [dataset for dataset in datasets if dataset.meaningful][:4]
    if not datasets:
        raise ValueError('No meaningful dashboard data is available.')
    _visual_analytics__load_matplotlib()
    fig, axes = _visual_analytics__plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    fig.suptitle('Smart Task Visual Dashboard', fontsize=22, fontweight='bold', color='#172B4D')
    for ax, dataset in zip(axes.flat, datasets):
        labels = [label for label, _ in dataset.series]
        values = [value for _, value in dataset.series]
        bars = ax.bar(labels, values, color=visual_analytics.COLORS[:len(values)], width=0.62)
        ax.set_ylim(bottom=0, top=max(values) * 1.25 if max(values) else 1)
        ax.grid(axis='y', color='#DFE1E6', linewidth=0.7)
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.tick_params(axis='x', rotation=22, labelsize=8)
        visual_analytics._style_axis(ax, dataset.title, dataset.scope)
    for ax in axes.flat[len(datasets):]:
        ax.axis('off')
    return visual_analytics._finish_png(fig)
visual_analytics.render_dashboard_png = _visual_analytics__render_dashboard_png
def _visual_analytics__render_dashboard(datasets) -> str:
    """Render a compact two-column dashboard from independent exact datasets."""
    datasets = [dataset for dataset in datasets if dataset.meaningful][:4]
    if not datasets:
        raise ValueError('No meaningful dashboard data is available.')
    width, column_width = (1400, 630)
    panels, bottoms = ([], [115, 115])
    colors = ('#4C78A8', '#E45756', '#72B7B2', '#F2CF5B')
    for index, dataset in enumerate(datasets):
        column = index % 2
        x, y = (55 + column * 685, bottoms[column])
        panel, height = visual_analytics._panel(dataset.title, dataset.series, x, y, column_width, color=colors[index])
        panels.append(panel)
        bottoms[column] += height + 30
    height = max(bottoms) + 30
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Task analytics dashboard"><style>.title{{font:700 34px Arial,sans-serif;fill:#172B4D}}.panel{{font:700 21px Arial,sans-serif;fill:#172B4D}}.label{{font:17px Arial,sans-serif;fill:#344563}}.value{{font:700 17px Arial,sans-serif;fill:#172B4D}}</style><rect width="100%" height="100%" fill="#F4F6F8"/><text x="55" y="58" class="title">Smart Task Visual Dashboard</text>' + ''.join(panels) + '</svg>'
visual_analytics.render_dashboard = _visual_analytics__render_dashboard


# transcription.py
'Safe speech transcription for Slack media with built-in provider selection.'
from dataclasses import dataclass as _transcription__dataclass
transcription.dataclass = _transcription__dataclass
import atexit as _transcription__atexit
transcription.atexit = _transcription__atexit
import importlib.util as _transcription__importlib
transcription.importlib = _transcription__importlib
import json as _transcription__json
transcription.json = _transcription__json
import logging as _transcription__logging
transcription.logging = _transcription__logging
import math as _transcription__math
transcription.math = _transcription__math
import multiprocessing as _transcription__multiprocessing
transcription.multiprocessing = _transcription__multiprocessing
import os as _transcription__os
transcription.os = _transcription__os
from pathlib import Path as _transcription__Path
transcription.Path = _transcription__Path
import shlex as _transcription__shlex
transcription.shlex = _transcription__shlex
import shutil as _transcription__shutil
transcription.shutil = _transcription__shutil
import subprocess as _transcription__subprocess
transcription.subprocess = _transcription__subprocess
import sys as _transcription__sys
transcription.sys = _transcription__sys
import tempfile as _transcription__tempfile
transcription.tempfile = _transcription__tempfile
import threading as _transcription__threading
transcription.threading = _transcription__threading
import time as _transcription__time
transcription.time = _transcription__time
import uuid as _transcription__uuid
transcription.uuid = _transcription__uuid
from urllib.error import HTTPError as _transcription__HTTPError, URLError as _transcription__URLError
transcription.HTTPError = _transcription__HTTPError
transcription.URLError = _transcription__URLError
from urllib.request import Request as _transcription__Request, urlopen as _transcription__urlopen
transcription.Request = _transcription__Request
transcription.urlopen = _transcription__urlopen
_transcription__logger = _transcription__logging.getLogger('transcription')
transcription.logger = _transcription__logger
class _transcription__TranscriptionError(RuntimeError):

    def __init__(self, message, stage='transcription'):
        super().__init__(message)
        self.stage = stage
transcription.TranscriptionError = _transcription__TranscriptionError
@_transcription__dataclass(frozen=True)
class _transcription__Transcript:
    text: str
    chunks: int
    duration_seconds: float | None = None
    confidence: float | None = None
transcription.Transcript = _transcription__Transcript
_transcription___TEXT_OUTPUT_SUFFIXES = {'.txt', '.vtt', '.srt', '.tsv', '.json'}
transcription._TEXT_OUTPUT_SUFFIXES = _transcription___TEXT_OUTPUT_SUFFIXES
_transcription___OPENAI_TRANSCRIPTION_URL = 'https://api.openai.com/v1/audio/transcriptions'
transcription._OPENAI_TRANSCRIPTION_URL = _transcription___OPENAI_TRANSCRIPTION_URL
from src.prompts import DEFAULT_WHISPER_PROMPT as _transcription___DEFAULT_WHISPER_PROMPT
transcription._DEFAULT_WHISPER_PROMPT = _transcription___DEFAULT_WHISPER_PROMPT
_transcription___local_provider = None
transcription._local_provider = _transcription___local_provider
_transcription___local_provider_key = None
transcription._local_provider_key = _transcription___local_provider_key
_transcription___local_provider_lock = _transcription__threading.Lock()
transcription._local_provider_lock = _transcription___local_provider_lock
def _transcription___local_whisper_worker(connection, model_name, device, language, prompt):
    """Own one Whisper model for the lifetime of a killable worker process."""
    try:
        import whisper
        model = whisper.load_model(model_name, device=device)
        while True:
            request = connection.recv()
            if request is None:
                return
            request_id, audio_path = request
            try:
                options = {'verbose': None, 'temperature': 0, 'condition_on_previous_text': False, 'fp16': False}
                if language:
                    options['language'] = language
                if prompt:
                    options['initial_prompt'] = prompt
                result = model.transcribe(audio_path, **options)
                segments = result.get('segments') or []
                log_probabilities = [float(segment['avg_logprob']) for segment in segments if segment.get('avg_logprob') is not None]
                confidence = _transcription__math.exp(sum(log_probabilities) / len(log_probabilities)) if log_probabilities else None
                connection.send((request_id, 'ok', {'text': str(result.get('text') or '').strip(), 'confidence': confidence}))
            except BaseException as exc:
                connection.send((request_id, 'error', type(exc).__name__))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        return
    except BaseException as exc:
        try:
            connection.send(('startup', 'error', type(exc).__name__))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()
transcription._local_whisper_worker = _transcription___local_whisper_worker
class _transcription__LocalWhisperProvider:
    """Persistent, serialized Whisper model hosted in a terminable process."""

    def __init__(self, model_name, device='cpu', language=None, prompt=None):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.prompt = prompt
        self._process = None
        self._connection = None
        self.last_confidence = None
        self._lock = _transcription__threading.Lock()

    def _start(self):
        if self._process is not None and self._process.is_alive():
            return
        self.close()
        context = _transcription__multiprocessing.get_context('spawn')
        parent, child = context.Pipe(duplex=True)
        process = context.Process(target=transcription._local_whisper_worker, args=(child, self.model_name, self.device, self.language, self.prompt), name='slack-list-whisper', daemon=True)
        process.start()
        child.close()
        self._connection, self._process = (parent, process)
        transcription.logger.info('transcription_provider_initialized provider=whisper model=%s device=%s pid=%s', self.model_name, self.device, process.pid)

    def transcribe(self, path, timeout):
        request_id = _transcription__uuid.uuid4().hex
        with self._lock:
            self._start()
            try:
                self._connection.send((request_id, str(path)))
                if not self._connection.poll(timeout):
                    transcription.logger.warning('transcription_provider_timeout provider=whisper model=%s timeout_seconds=%s', self.model_name, timeout)
                    self.close(force=True)
                    raise transcription.TranscriptionError('The transcription provider timed out.', stage='provider_timeout')
                observed_id, status, value = self._connection.recv()
                if observed_id not in {request_id, 'startup'} or status != 'ok':
                    transcription.logger.warning('transcription_provider_failed provider=whisper model=%s error_type=%s', self.model_name, value)
                    self.close(force=True)
                    raise transcription.TranscriptionError('The local transcription provider failed.', stage='provider')
                self.last_confidence = value.get('confidence') if isinstance(value, dict) else None
                text = value.get('text') if isinstance(value, dict) else value
                if not str(text or '').strip():
                    raise transcription.TranscriptionError('The transcription provider returned an empty transcript.', stage='provider_output')
                return str(text).strip()
            except KeyboardInterrupt:
                self.close(force=True)
                raise
            except (EOFError, BrokenPipeError, OSError) as exc:
                self.close(force=True)
                raise transcription.TranscriptionError('The local transcription provider stopped unexpectedly.', stage='provider') from exc

    def close(self, force=False):
        process, connection = (self._process, self._connection)
        self._process = self._connection = None
        if connection is not None:
            if process is not None and process.is_alive() and (not force):
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            connection.close()
        if process is not None:
            process.join(timeout=2 if not force else 0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive() and hasattr(process, 'kill'):
                process.kill()
                process.join(timeout=1)
transcription.LocalWhisperProvider = _transcription__LocalWhisperProvider
def _transcription___local_whisper_available():
    return _transcription__importlib.util.find_spec('whisper') is not None
transcription._local_whisper_available = _transcription___local_whisper_available
def _transcription___local_whisper_provider():
    global _local_provider, _local_provider_key
    model = (_transcription__os.getenv('STT_MODEL') or _transcription__os.getenv('MEDIA_WHISPER_MODEL') or 'tiny.en').strip()
    device = _transcription__os.getenv('STT_DEVICE', 'cpu').strip() or 'cpu'
    language = _transcription__os.getenv('STT_LANGUAGE', '').strip() or ('en' if model.endswith('.en') else None)
    prompt = _transcription__os.getenv('STT_PROMPT', transcription._DEFAULT_WHISPER_PROMPT).strip()
    key = (model, device, language, prompt)
    with transcription._local_provider_lock:
        if transcription._local_provider is None or transcription._local_provider_key != key:
            if transcription._local_provider is not None:
                transcription._local_provider.close()
            transcription._local_provider = transcription.LocalWhisperProvider(model, device, language, prompt)
            transcription._local_provider_key = key
        return transcription._local_provider
transcription._local_whisper_provider = _transcription___local_whisper_provider
def _transcription__shutdown_transcription_provider():
    global _local_provider, _local_provider_key
    with transcription._local_provider_lock:
        if transcription._local_provider is not None:
            transcription._local_provider.close()
        transcription._local_provider = transcription._local_provider_key = None
transcription.shutdown_transcription_provider = _transcription__shutdown_transcription_provider
_transcription__atexit.register(transcription.shutdown_transcription_provider)
def _transcription___safe_process_detail(value, hidden_paths=()):
    detail = str(value or '').strip()
    for path in sorted({str(value) for value in hidden_paths if value}, key=len, reverse=True):
        detail = detail.replace(path, '<temporary path>')
    return detail[-500:]
transcription._safe_process_detail = _transcription___safe_process_detail
def _transcription___run(args, timeout=120, *, cwd=None, label='Media processing', hidden_paths=(), stage='media_processing'):
    """Run one trusted argv without a shell and normalize process failures."""
    try:
        result = _transcription__subprocess.run(list(args), check=False, capture_output=True, text=True, timeout=timeout, cwd=str(cwd) if cwd else None)
    except FileNotFoundError as exc:
        executable = _transcription__Path(str(args[0])).name
        if stage == 'provider':
            message = f'Transcription provider executable {executable!r} is not installed.'
        else:
            message = f'Required media tool {executable!r} is not installed.'
        raise transcription.TranscriptionError(message, stage=stage) from exc
    except _transcription__subprocess.TimeoutExpired as exc:
        raise transcription.TranscriptionError(f'{label} timed out.', stage=stage) from exc
    if result.returncode:
        detail = transcription._safe_process_detail(result.stderr or result.stdout, hidden_paths)
        if detail:
            raise transcription.TranscriptionError(f'{label} failed: {detail}', stage=stage)
        raise transcription.TranscriptionError(f'{label} failed with exit code {result.returncode}.', stage=stage)
    return result
transcription._run = _transcription___run
def _transcription___duration(path):
    result = transcription._run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], hidden_paths=(path.parent, path), label='ffprobe', stage='ffprobe')
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise transcription.TranscriptionError('ffprobe did not return a valid media duration.', stage='ffprobe')
transcription._duration = _transcription___duration
def _transcription___has_audio(path):
    result = transcription._run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', str(path)], hidden_paths=(path.parent, path), label='ffprobe', stage='ffprobe')
    return bool(result.stdout.strip())
transcription._has_audio = _transcription___has_audio
def _transcription___provider_argv(path, command, output_dir):
    try:
        template = _transcription__shlex.split(command)
    except ValueError as exc:
        raise transcription.TranscriptionError('MEDIA_TRANSCRIPTION_COMMAND contains invalid quoting.', stage='configuration') from exc
    if not template or not any(('{input}' in part for part in template)):
        raise transcription.TranscriptionError('MEDIA_TRANSCRIPTION_COMMAND must contain {input}.', stage='configuration')
    argv = []
    index = 0
    while index < len(template):
        part = template[index]
        if part in {'--output_dir', '--output-dir'}:
            argv.extend((part, str(output_dir)))
            index += 2
            continue
        if part.startswith('--output_dir=') or part.startswith('--output-dir='):
            argv.append(part.split('=', 1)[0] + '=' + str(output_dir))
            index += 1
            continue
        argv.append(part.replace('{input}', str(path)).replace('{output_dir}', str(output_dir)))
        index += 1
    return argv
transcription._provider_argv = _transcription___provider_argv
def _transcription___provider_output_format(argv):
    """Return a declared textual output format without assuming a provider."""
    for index, part in enumerate(argv):
        if part in {'--output_format', '--output-format'} and index + 1 < len(argv):
            return str(argv[index + 1]).casefold().lstrip('.')
        if part.startswith('--output_format=') or part.startswith('--output-format='):
            return part.split('=', 1)[1].casefold().lstrip('.')
    return None
transcription._provider_output_format = _transcription___provider_output_format
def _transcription___resolve_provider_executable(argv):
    """Find a bare provider command on PATH or beside the running Python."""
    if not argv:
        return argv
    executable = str(argv[0])
    if _transcription__os.sep in executable or (_transcription__os.altsep and _transcription__os.altsep in executable):
        return argv
    if _transcription__shutil.which(executable):
        return argv
    virtualenv_executable = _transcription__Path(_transcription__sys.executable).parent / executable
    if virtualenv_executable.is_file() and _transcription__os.access(virtualenv_executable, _transcription__os.X_OK):
        return [str(virtualenv_executable), *argv[1:]]
    return argv
transcription._resolve_provider_executable = _transcription___resolve_provider_executable
def _transcription___output_snapshot(directory):
    snapshot = {}
    if not directory.exists():
        return snapshot
    for candidate in directory.iterdir():
        if candidate.is_file() and candidate.suffix.casefold() in transcription._TEXT_OUTPUT_SUFFIXES:
            stat = candidate.stat()
            snapshot[candidate] = (stat.st_mtime_ns, stat.st_size)
    return snapshot
transcription._output_snapshot = _transcription___output_snapshot
def _transcription___read_transcript_file(path):
    try:
        raw = path.read_text(encoding='utf-8-sig')
    except (OSError, UnicodeError) as exc:
        raise transcription.TranscriptionError('The transcription provider produced an unreadable transcript file.', stage='provider_output') from exc
    if path.suffix.casefold() == '.json':
        try:
            payload = _transcription__json.loads(raw)
        except _transcription__json.JSONDecodeError as exc:
            raise transcription.TranscriptionError('The transcription provider produced malformed transcript JSON.', stage='provider_output') from exc
        if isinstance(payload, dict):
            raw = payload.get('text') or payload.get('transcript') or ''
        elif isinstance(payload, str):
            raw = payload
        else:
            raw = ''
    return str(raw).strip()
transcription._read_transcript_file = _transcription___read_transcript_file
def _transcription___invoke_provider(path, command, timeout, *, file_id='unknown', chunk_index=1, chunk_count=1):
    """Return stdout or a newly generated transcript file, then clean it up."""
    with _transcription__tempfile.TemporaryDirectory(prefix='provider-', dir=path.parent) as work_dir_value:
        work_dir = _transcription__Path(work_dir_value)
        before_parent = transcription._output_snapshot(path.parent)
        argv = transcription._resolve_provider_executable(transcription._provider_argv(path, command, work_dir))
        output_format = transcription._provider_output_format(argv)
        candidates = []
        generated_candidates = []
        transcription.logger.info('transcription_provider_started file_id=%s chunk_index=%d chunk_count=%d', redact(file_id), chunk_index, chunk_count)
        try:
            result = transcription._run(argv, timeout=timeout, cwd=work_dir, label='The transcription provider', hidden_paths=(path.parent, path, work_dir), stage='provider')
            for directory in (work_dir, path.parent):
                for candidate in directory.iterdir():
                    if not candidate.is_file() or candidate.suffix.casefold() not in transcription._TEXT_OUTPUT_SUFFIXES:
                        continue
                    stat = candidate.stat()
                    signature = (stat.st_mtime_ns, stat.st_size)
                    if directory == work_dir or (before_parent.get(candidate) != signature and candidate.stem == path.stem):
                        candidates.append(candidate)
            generated_candidates = list(candidates)
            declared_file_output = output_format in {suffix.lstrip('.') for suffix in transcription._TEXT_OUTPUT_SUFFIXES}
            if declared_file_output:
                expected = [directory / f'{path.stem}.{output_format}' for directory in (work_dir, path.parent)]
                candidates = [candidate for candidate in expected if candidate in candidates]
            else:
                same_stem = [candidate for candidate in candidates if candidate.stem == path.stem]
                if same_stem:
                    candidates = same_stem
            if declared_file_output and (not candidates):
                raise transcription.TranscriptionError(f'The transcription provider did not create the expected .{output_format} transcript file.', stage='provider_output')
            texts = []
            for candidate in sorted(set(candidates), key=lambda value: (value.name, str(value))):
                value = transcription._read_transcript_file(candidate)
                if value:
                    texts.append(value)
            text = '\n'.join(texts).strip()
            if not text and (not declared_file_output):
                text = result.stdout.strip()
            if not text:
                raise transcription.TranscriptionError('The transcription provider returned an empty transcript.', stage='provider_output')
            transcription.logger.info('transcription_provider_completed file_id=%s chunk_index=%d chunk_count=%d output_mode=%s transcript_chars=%d', redact(file_id), chunk_index, chunk_count, 'file' if texts else 'stdout', len(text))
            return text
        except transcription.TranscriptionError as exc:
            transcription.logger.warning('transcription_provider_failed file_id=%s chunk_index=%d chunk_count=%d stage=%s error_type=%s message=%s', redact(file_id), chunk_index, chunk_count, exc.stage, type(exc).__name__, redact(exc))
            raise
        finally:
            cleanup_candidates = set(generated_candidates)
            cleanup_candidates.update((candidate for candidate in transcription._output_snapshot(path.parent) if candidate not in before_parent and candidate.stem == path.stem))
            for candidate in cleanup_candidates:
                if candidate.parent == path.parent and candidate not in before_parent:
                    try:
                        candidate.unlink()
                    except FileNotFoundError:
                        pass
transcription._invoke_provider = _transcription___invoke_provider
def _transcription___multipart_body(path, model):
    boundary = '----slack-list-' + _transcription__uuid.uuid4().hex
    body = []
    for name, value in (('model', model),):
        body.extend((f'--{boundary}\r\n'.encode(), f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(), str(value).encode(), b'\r\n'))
    body.extend((f'--{boundary}\r\n'.encode(), b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n', b'Content-Type: audio/wav\r\n\r\n', path.read_bytes(), b'\r\n', f'--{boundary}--\r\n'.encode()))
    return (b''.join(body), boundary)
transcription._multipart_body = _transcription___multipart_body
def _transcription___invoke_openai(path, api_key, model, timeout, *, file_id='unknown', chunk_index=1, chunk_count=1, opener=_transcription__urlopen, sleeper=None):
    body, boundary = transcription._multipart_body(path, model)
    request = _transcription__Request(transcription._OPENAI_TRANSCRIPTION_URL, data=body, method='POST', headers={'Authorization': f'Bearer {api_key}', 'Content-Type': f'multipart/form-data; boundary={boundary}'})
    sleeper = sleeper or (lambda seconds: __import__('time').sleep(seconds))
    transcription.logger.info('transcription_provider_started provider=openai file_id=%s chunk_index=%d chunk_count=%d model=%s', redact(file_id), chunk_index, chunk_count, model)
    for attempt in range(2):
        try:
            with opener(request, timeout=timeout) as response:
                payload = _transcription__json.loads(response.read().decode('utf-8'))
            break
        except _transcription__HTTPError as exc:
            transient = exc.code == 429 or exc.code >= 500
            if transient and attempt == 0:
                transcription.logger.warning('transcription_provider_retry provider=openai status=%s attempt=1/2', exc.code)
                sleeper(1)
                continue
            message = 'The configured transcription credential was rejected.' if exc.code in {401, 403} else 'The transcription provider is temporarily unavailable.' if transient else 'The transcription provider rejected the media request.'
            raise transcription.TranscriptionError(message, stage='provider') from exc
        except (TimeoutError, _transcription__URLError) as exc:
            if attempt == 0:
                transcription.logger.warning('transcription_provider_retry provider=openai status=network attempt=1/2')
                sleeper(1)
                continue
            raise transcription.TranscriptionError('The transcription provider could not be reached or timed out.', stage='provider') from exc
        except (OSError, UnicodeError, _transcription__json.JSONDecodeError) as exc:
            raise transcription.TranscriptionError('The transcription provider returned an unreadable response.', stage='provider_output') from exc
    text = str(payload.get('text') or '').strip() if isinstance(payload, dict) else ''
    if not text:
        raise transcription.TranscriptionError('The transcription provider returned an empty transcript.', stage='provider_output')
    transcription.logger.info('transcription_provider_completed provider=openai file_id=%s chunk_index=%d chunk_count=%d transcript_chars=%d', redact(file_id), chunk_index, chunk_count, len(text))
    return text
transcription._invoke_openai = _transcription___invoke_openai
def _transcription___transcription_backend(command=None, provider=None):
    if command:
        return ('command', command)
    configured = _transcription__os.getenv('MEDIA_TRANSCRIPTION_COMMAND', '').strip()
    selected = str(provider or _transcription__os.getenv('MEDIA_TRANSCRIPTION_PROVIDER', 'auto')).strip().casefold()
    if selected in {'', 'auto'}:
        if configured:
            return ('command', configured)
        if _transcription__os.getenv('OPENAI_API_KEY', '').strip():
            return ('openai', None)
        if transcription._local_whisper_available():
            return ('whisper', None)
        raise transcription.TranscriptionError('Audio/video transcription is not configured. Configure OPENAI_API_KEY, select the Whisper backend, or set MEDIA_TRANSCRIPTION_COMMAND.', stage='configuration')
    if selected == 'openai':
        if not _transcription__os.getenv('OPENAI_API_KEY', '').strip():
            raise transcription.TranscriptionError('OpenAI transcription requires OPENAI_API_KEY.', stage='configuration')
        return ('openai', None)
    if selected in {'command', 'custom'}:
        if not configured:
            raise transcription.TranscriptionError('The command transcription backend requires MEDIA_TRANSCRIPTION_COMMAND.', stage='configuration')
        return ('command', configured)
    if selected in {'whisper', 'local', 'local-whisper'}:
        if not transcription._local_whisper_available():
            raise transcription.TranscriptionError('The local Whisper transcription backend is not installed.', stage='configuration')
        return ('whisper', None)
    raise transcription.TranscriptionError('MEDIA_TRANSCRIPTION_PROVIDER must be auto, openai, whisper, or command.', stage='configuration')
transcription._transcription_backend = _transcription___transcription_backend
def _transcription__transcribe_bytes(data: bytes, media_kind: str, mimetype: str='', command=None, chunk_seconds=None, max_seconds=None, timeout=None, file_id='unknown', provider=None):
    if not data:
        raise transcription.TranscriptionError('The media file is empty.', stage='validation')
    backend, command = transcription._transcription_backend(command, provider)
    transcription.logger.info('transcription_provider_selected provider=%s file_id=%s', backend, redact(file_id))
    try:
        chunk_seconds = int(chunk_seconds or _transcription__os.getenv('MEDIA_TRANSCRIPTION_CHUNK_SECONDS', '600'))
        max_seconds = int(max_seconds or _transcription__os.getenv('MEDIA_TRANSCRIPTION_MAX_SECONDS', '14400'))
        timeout = int(timeout or _transcription__os.getenv('STT_TIMEOUT_SECONDS', _transcription__os.getenv('MEDIA_TRANSCRIPTION_TIMEOUT_SECONDS', '60')))
    except (TypeError, ValueError) as exc:
        raise transcription.TranscriptionError('Media transcription limits must be valid whole numbers.', stage='configuration') from exc
    if chunk_seconds <= 0 or max_seconds <= 0 or timeout <= 0:
        raise transcription.TranscriptionError('Media transcription limits must be greater than zero.', stage='configuration')
    if not _transcription__shutil.which('ffprobe'):
        raise transcription.TranscriptionError("Required media tool 'ffprobe' is not installed.", stage='ffprobe')
    if not _transcription__shutil.which('ffmpeg'):
        raise transcription.TranscriptionError("Required media tool 'ffmpeg' is not installed.", stage='ffmpeg')
    suffix = '.mp4' if media_kind == 'video' else '.audio'
    with _transcription__tempfile.TemporaryDirectory(prefix='slack-media-') as temp_dir:
        source = _transcription__Path(temp_dir) / ('source' + suffix)
        source.write_bytes(data)
        if not transcription._has_audio(source):
            raise transcription.TranscriptionError('The media contains no usable audio track.', stage='audio_validation')
        duration = transcription._duration(source)
        if duration and duration > max_seconds:
            raise transcription.TranscriptionError(f'The recording is longer than the configured {max_seconds // 60}-minute limit.', stage='duration_validation')
        transcription.logger.info('media_validation_completed file_id=%s media_type=%s duration_seconds=%.3f', redact(file_id), media_kind, duration or 0.0)
        pattern = _transcription__Path(temp_dir) / 'chunk-%04d.wav'
        transcription._run(['ffmpeg', '-v', 'error', '-i', str(source), '-vn', '-ac', '1', '-ar', '16000', '-f', 'segment', '-segment_time', str(chunk_seconds), str(pattern)], timeout=max(timeout, 120), hidden_paths=(temp_dir, source), label='FFmpeg audio normalization', stage='ffmpeg')
        chunks = sorted(_transcription__Path(temp_dir).glob('chunk-*.wav'))
        if not chunks:
            raise transcription.TranscriptionError('No usable speech audio could be extracted from the media.', stage='ffmpeg')
        if backend == 'openai':
            api_key = _transcription__os.getenv('OPENAI_API_KEY', '').strip()
            model = _transcription__os.getenv('MEDIA_TRANSCRIPTION_MODEL', 'gpt-4o-mini-transcribe').strip()
            transcripts = [transcription._invoke_openai(chunk, api_key, model, timeout, file_id=file_id, chunk_index=index, chunk_count=len(chunks)) for index, chunk in enumerate(chunks, 1)]
        elif backend == 'whisper':
            provider_instance = transcription._local_whisper_provider()
            transcripts = []
            confidences = []
            for index, chunk in enumerate(chunks, 1):
                provider_started = _transcription__time.monotonic()
                transcription.logger.info('transcription_provider_started provider=whisper model=%s file_id=%s chunk_index=%d chunk_count=%d', provider_instance.model_name, redact(file_id), index, len(chunks))
                value = provider_instance.transcribe(chunk, timeout)
                transcription.logger.info('transcription_provider_completed provider=whisper model=%s file_id=%s chunk_index=%d chunk_count=%d transcript_chars=%d latency_ms=%d', provider_instance.model_name, redact(file_id), index, len(chunks), len(value), int((_transcription__time.monotonic() - provider_started) * 1000))
                transcripts.append(value)
                if provider_instance.last_confidence is not None:
                    confidences.append(provider_instance.last_confidence)
        else:
            transcripts = [transcription._invoke_provider(chunk, command, timeout, file_id=file_id, chunk_index=index, chunk_count=len(chunks)) for index, chunk in enumerate(chunks, 1)]
        combined = '\n'.join((part for part in transcripts if part.strip())).strip()
        if not combined:
            raise transcription.TranscriptionError('The recording did not produce usable transcript text.', stage='provider_output')
        confidence = min(confidences) if backend == 'whisper' and confidences else None
        return transcription.Transcript(combined, len(chunks), duration, confidence)
transcription.transcribe_bytes = _transcription__transcribe_bytes


# content_ingestion.py
'Detect and obtain text/transcript/audio/video content from Slack events.'
from dataclasses import dataclass as _content_ingestion__dataclass
content_ingestion.dataclass = _content_ingestion__dataclass
import logging as _content_ingestion__logging
content_ingestion.logging = _content_ingestion__logging
import os as _content_ingestion__os
content_ingestion.os = _content_ingestion__os
import re as _content_ingestion__re
content_ingestion.re = _content_ingestion__re
import time as _content_ingestion__time
content_ingestion.time = _content_ingestion__time
from urllib.parse import urlparse as _content_ingestion__urlparse
content_ingestion.urlparse = _content_ingestion__urlparse
from urllib.error import HTTPError as _content_ingestion__HTTPError, URLError as _content_ingestion__URLError
content_ingestion.HTTPError = _content_ingestion__HTTPError
content_ingestion.URLError = _content_ingestion__URLError
from urllib.request import HTTPRedirectHandler as _content_ingestion__HTTPRedirectHandler, Request as _content_ingestion__Request, build_opener as _content_ingestion__build_opener, urlopen as _content_ingestion__urlopen
content_ingestion.HTTPRedirectHandler = _content_ingestion__HTTPRedirectHandler
content_ingestion.Request = _content_ingestion__Request
content_ingestion.build_opener = _content_ingestion__build_opener
content_ingestion.urlopen = _content_ingestion__urlopen
from slack_sdk.errors import SlackApiError as _content_ingestion__SlackApiError
content_ingestion.SlackApiError = _content_ingestion__SlackApiError
content_ingestion.transcription = transcription
_content_ingestion__logger = _content_ingestion__logging.getLogger('content_ingestion')
content_ingestion.logger = _content_ingestion__logger
class _content_ingestion__ContentError(ValueError):
    pass
content_ingestion.ContentError = _content_ingestion__ContentError
class _content_ingestion__ContentAuthorizationError(content_ingestion.ContentError):
    pass
content_ingestion.ContentAuthorizationError = _content_ingestion__ContentAuthorizationError
class _content_ingestion__TemporaryContentError(content_ingestion.ContentError):
    pass
content_ingestion.TemporaryContentError = _content_ingestion__TemporaryContentError
@_content_ingestion__dataclass(frozen=True)
class _content_ingestion__IngestedContent:
    text: str
    source_type: str
    source_reference: str
    chunks: int = 1
content_ingestion.IngestedContent = _content_ingestion__IngestedContent
@_content_ingestion__dataclass(frozen=True)
class _content_ingestion__NormalizedRequest:
    source_type: str
    raw_input: str
    transcript: str | None
    normalized_text: str
    requester_id: str | None
    channel_id: str | None
    thread_context: str | None
    source_file: str | None = None

    @property
    def text(self):
        return self.normalized_text
content_ingestion.NormalizedRequest = _content_ingestion__NormalizedRequest
def _content_ingestion__normalize_text_request(text, *, requester_id=None, channel_id=None, thread_context=None):
    normalized = _content_ingestion__re.sub('[ \\t]+', ' ', str(text or '').replace('\x00', ''))
    normalized = _content_ingestion__re.sub('\\n{3,}', '\n\n', normalized).strip()
    if not normalized:
        raise content_ingestion.ContentError('The request contains no usable text.')
    return content_ingestion.NormalizedRequest('text', str(text or ''), None, normalized, requester_id, channel_id, thread_context)
content_ingestion.normalize_text_request = _content_ingestion__normalize_text_request
def _content_ingestion__normalized_request(content, *, requester_id=None, channel_id=None, thread_context=None):
    normalized = content_ingestion.normalize_transcript(content.text)
    if not normalized:
        raise content_ingestion.ContentError('The transcript is empty or contains no usable text.')
    content_ingestion.logger.info('transcript_normalized source=%s file_id=%s transcript_chars=%d', content.source_type, redact(content.source_reference), len(normalized))
    return content_ingestion.NormalizedRequest(content.source_type, content.source_reference, normalized, normalized, requester_id, channel_id, thread_context, content.source_reference)
content_ingestion.normalized_request = _content_ingestion__normalized_request
@_content_ingestion__dataclass(frozen=True)
class _content_ingestion__RequestRoute:
    """Immutable source classification made before either processing pipeline runs."""
    route: str
    source: str
    source_types: tuple[str, ...] = ()

    @property
    def is_shared_content(self):
        return self.route in {'media', 'transcript'}
content_ingestion.RequestRoute = _content_ingestion__RequestRoute
_content_ingestion___MEDIA_REQUEST = _content_ingestion__re.compile('\\b(?:extract|identify|find|capture|derive|turn|convert|create|add)\\b.*\\b(?:action\\s+items?|tasks?|todos?)\\b|\\b(?:action\\s+items?|tasks?|todos?)\\b.*\\b(?:from|in)\\b.*\\b(?:audio|video|recording|meeting|transcript|attachment|file)\\b', _content_ingestion__re.I | _content_ingestion__re.S)
content_ingestion._MEDIA_REQUEST = _content_ingestion___MEDIA_REQUEST
_content_ingestion___PREVIEW = _content_ingestion__re.compile("\\b(?:preview|review\\s+(?:first|only)|draft\\s+only|extract\\s+only|do\\s+not|don't|without)\\b.*\\b(?:create|add|change|mutate)\\b", _content_ingestion__re.I | _content_ingestion__re.S)
content_ingestion._PREVIEW = _content_ingestion___PREVIEW
def _content_ingestion__extraction_requested(text):
    return bool(content_ingestion._MEDIA_REQUEST.search(str(text or '')))
content_ingestion.extraction_requested = _content_ingestion__extraction_requested
def _content_ingestion__preview_requested(text):
    return bool(content_ingestion._PREVIEW.search(str(text or '')))
content_ingestion.preview_requested = _content_ingestion__preview_requested
def _content_ingestion__explicit_transcript_payload(text):
    """Detect delimited transcript content, never a mere transcript keyword."""
    value = str(text or '')
    if _content_ingestion__re.match('^\\s*transcript\\s*:\\s*\\S', value, _content_ingestion__re.I | _content_ingestion__re.S):
        return True
    if '\n' not in value:
        return False
    header, body = value.split('\n', 1)
    return bool(_content_ingestion__re.search('\\btranscript\\b[^:\\n]{0,80}:\\s*$', header, _content_ingestion__re.I) and body.strip())
content_ingestion.explicit_transcript_payload = _content_ingestion__explicit_transcript_payload
def _content_ingestion__classify_request(text, files=(), attachments=()):
    """Choose the text, media, or transcript pipeline from Slack event data.

    Instruction keywords and parser outcomes deliberately have no influence on
    this boundary. A Slack file entry is shared content even if its event stub
    needs ``files.info`` before its exact type is known.
    """
    kinds = []
    has_file = False
    for value in files or []:
        if not isinstance(value, dict):
            continue
        has_file = has_file or bool(value.get('id') or value.get('filetype') or value.get('mimetype') or value.get('mode'))
        kind = content_ingestion.file_kind(value)
        if kind != 'unsupported':
            kinds.append(kind)
    for value in attachments or []:
        if not isinstance(value, dict):
            continue
        kind = content_ingestion.file_kind(value)
        if kind != 'unsupported':
            has_file = True
            kinds.append(kind)
    unique = tuple(dict.fromkeys(kinds))
    if has_file:
        source = unique[0] if len(unique) == 1 else 'mixed' if unique else 'file'
        route = 'transcript' if unique and set(unique) == {'transcript'} else 'media'
        return content_ingestion.RequestRoute(route, source, unique)
    if content_ingestion.explicit_transcript_payload(text):
        return content_ingestion.RequestRoute('transcript', 'transcript', ('transcript',))
    return content_ingestion.RequestRoute('text', 'text')
content_ingestion.classify_request = _content_ingestion__classify_request
def _content_ingestion__should_ingest(text, files=(), attachments=()):
    """Compatibility predicate backed by the source-first request router."""
    return content_ingestion.classify_request(text, files, attachments).is_shared_content
content_ingestion.should_ingest = _content_ingestion__should_ingest
def _content_ingestion__file_kind(file_info):
    mime = str(file_info.get('mimetype') or '').split(';', 1)[0].strip().casefold()
    filetype = str(file_info.get('filetype') or '').strip().casefold().lstrip('.')
    mode = str(file_info.get('mode') or '').casefold()
    name = str(file_info.get('name') or file_info.get('title') or '')
    extension = _content_ingestion__os.path.splitext(name)[1].casefold().lstrip('.')
    generic_mime = mime in {'', 'application/octet-stream', 'binary/octet-stream', 'application/binary'}
    if not filetype and generic_mime:
        filetype = extension
    if mime.startswith('audio/') or mode == 'audio' or filetype in {'mp3', 'm4a', 'wav', 'ogg', 'opus', 'aac', 'flac'}:
        return 'audio'
    if mime.startswith('video/') or mode == 'video' or filetype in {'mp4', 'mov', 'webm', 'mkv', 'avi', 'mpeg'}:
        return 'video'
    if mime.startswith('text/') or filetype in {'txt', 'text', 'vtt', 'srt', 'transcript'}:
        return 'transcript'
    return 'unsupported'
content_ingestion.file_kind = _content_ingestion__file_kind
def _content_ingestion___allowed_url(url):
    parsed = _content_ingestion__urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname:
        return False
    allowed = {'slack.com', 'files.slack.com'}
    allowed.update((host.strip().casefold() for host in _content_ingestion__os.getenv('MEDIA_ALLOWED_HOSTS', '').split(',') if host.strip()))
    host = parsed.hostname.casefold()
    return any((host == value or host.endswith('.' + value) for value in allowed))
content_ingestion._allowed_url = _content_ingestion___allowed_url
def _content_ingestion___is_slack_url(url):
    host = (_content_ingestion__urlparse(str(url or '')).hostname or '').casefold()
    return host == 'slack.com' or host.endswith('.slack.com')
content_ingestion._is_slack_url = _content_ingestion___is_slack_url
def _content_ingestion___safe_file_metadata(file_info):
    return {'file_id': str(file_info.get('id') or 'unknown'), 'file_name': redact(file_info.get('name') or file_info.get('title') or 'unknown'), 'mimetype': str(file_info.get('mimetype') or 'unknown'), 'file_size': file_info.get('size') or 'unknown', 'file_access': str(file_info.get('file_access') or 'unknown'), 'has_private_url': bool(file_info.get('url_private_download') or file_info.get('url_private'))}
content_ingestion._safe_file_metadata = _content_ingestion___safe_file_metadata
def _content_ingestion__resolve_file_metadata(file_info, slack_client=None, attempts=3, sleeper=_content_ingestion__time.sleep):
    """Refresh an event file stub through files.info without changing identity."""
    current = dict(file_info or {})
    file_id = str(current.get('id') or '').strip()
    if not file_id or slack_client is None:
        content_ingestion.logger.info('file_metadata_resolved source=event %s', content_ingestion._safe_file_metadata(current))
        return current
    transient = {'internal_error', 'service_unavailable', 'request_timeout', 'ratelimited'}
    for attempt in range(1, attempts + 1):
        try:
            response = slack_client.files_info(file=file_id)
            fresh = dict(response.get('file') or {})
            if not fresh:
                raise content_ingestion.ContentError('Slack returned no metadata for the shared file.')
            current.update(fresh)
            content_ingestion.logger.info('file_metadata_resolved source=files.info %s', content_ingestion._safe_file_metadata(current))
            return current
        except _content_ingestion__SlackApiError as exc:
            error = str(exc.response.get('error') or 'slack_api_error')
            needed = str(exc.response.get('needed') or '')
            content_ingestion.logger.warning('file_metadata_failed file_id=%s error=%s needed_scope=%s attempt=%d/%d', file_id, error, needed or 'none', attempt, attempts)
            if error == 'missing_scope':
                raise content_ingestion.ContentAuthorizationError('The Slack app is missing the files:read OAuth scope required to read uploaded files. Add the bot scope and reinstall the app to the workspace.') from exc
            if error in {'invalid_auth', 'not_authed', 'token_expired', 'token_revoked'}:
                raise content_ingestion.ContentAuthorizationError('Slack could not authenticate access to the shared file.') from exc
            if error in {'access_denied', 'no_permission', 'not_visible'}:
                raise content_ingestion.ContentAuthorizationError('The Slack app is not allowed to access this file or conversation.') from exc
            if error in {'file_deleted', 'file_not_found'}:
                raise content_ingestion.ContentError('The shared Slack file no longer exists or is unavailable.') from exc
            if error in transient and attempt < attempts:
                sleeper(min(2 ** (attempt - 1), 4))
                continue
            if error in transient:
                raise content_ingestion.TemporaryContentError('Slack temporarily could not return the shared file metadata.') from exc
            log_exception(content_ingestion.logger, 'Slack file metadata request failed', exc, function='resolve_file_metadata', file_id=file_id, slack_error=error)
            raise content_ingestion.ContentError('Slack could not return metadata for the shared file.') from exc
        except content_ingestion.ContentError:
            raise
        except Exception as exc:
            log_exception(content_ingestion.logger, 'Slack file metadata request failed', exc, function='resolve_file_metadata', file_id=file_id, attempt=attempt)
            if attempt < attempts:
                sleeper(min(2 ** (attempt - 1), 4))
                continue
            raise content_ingestion.TemporaryContentError('Slack temporarily could not return the shared file metadata.') from exc
    raise content_ingestion.TemporaryContentError('Slack temporarily could not return the shared file metadata.')
content_ingestion.resolve_file_metadata = _content_ingestion__resolve_file_metadata
class _content_ingestion___SlackRedirectHandler(_content_ingestion__HTTPRedirectHandler):
    """Allow authenticated redirects only between Slack-owned HTTPS hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not content_ingestion._is_slack_url(newurl):
            raise content_ingestion.ContentAuthorizationError('Slack redirected the private file to an unapproved host.')
        return super().redirect_request(req, fp, code, msg, headers, newurl)
content_ingestion._SlackRedirectHandler = _content_ingestion___SlackRedirectHandler
def _content_ingestion__download(file_info, bot_token, max_bytes=None, attempts=3, sleeper=_content_ingestion__time.sleep, opener=None):
    inline = file_info.get('content')
    if isinstance(inline, str):
        return inline.encode('utf-8')
    if isinstance(inline, bytes):
        return inline
    url = file_info.get('url_private_download') or file_info.get('url_private') or file_info.get('url')
    if not url or not content_ingestion._allowed_url(url):
        raise content_ingestion.ContentError('The shared content is not available from an approved accessible URL.')
    max_bytes = int(max_bytes or _content_ingestion__os.getenv('MEDIA_MAX_BYTES', str(250 * 1024 * 1024)))
    is_slack = content_ingestion._is_slack_url(url)
    headers = {'Authorization': 'Bearer ' + bot_token} if bot_token and is_slack else {}
    if is_slack and (not bot_token):
        raise content_ingestion.ContentAuthorizationError('Slack file download requires the configured bot token.')
    opener = opener or _content_ingestion__build_opener(content_ingestion._SlackRedirectHandler()).open
    metadata = content_ingestion._safe_file_metadata(file_info)
    for attempt in range(1, attempts + 1):
        content_ingestion.logger.info('file_download_started file_id=%s attempt=%d/%d', metadata['file_id'], attempt, attempts)
        try:
            with opener(_content_ingestion__Request(url, headers=headers), timeout=60) as response:
                status = getattr(response, 'status', None)
                if status is None:
                    status = response.getcode()
                final_url = getattr(response, 'url', url)
                if is_slack and (not content_ingestion._is_slack_url(final_url)):
                    raise content_ingestion.ContentAuthorizationError('Slack redirected the private file to an unapproved host.')
                declared = response.headers.get('Content-Length')
                content_type = str(response.headers.get('Content-Type') or '').split(';', 1)[0]
                if declared and int(declared) > max_bytes:
                    raise content_ingestion.ContentError('The shared file exceeds the configured media size limit.')
                data = response.read(max_bytes + 1)
                if content_type.casefold() == 'text/html' and content_ingestion.file_kind(file_info) in {'audio', 'video'}:
                    raise content_ingestion.ContentAuthorizationError('Slack returned a sign-in page instead of the private media file. Verify the files:read bot scope and reinstall the app.')
                content_ingestion.logger.info('file_download_completed file_id=%s download_status=%s content_type=%s downloaded_bytes=%d', metadata['file_id'], status, content_type or 'unknown', len(data))
                break
        except content_ingestion.ContentError:
            raise
        except _content_ingestion__HTTPError as exc:
            content_ingestion.logger.warning('file_download_failed file_id=%s status=%s attempt=%d/%d', metadata['file_id'], exc.code, attempt, attempts)
            if exc.code in {401, 403}:
                raise content_ingestion.ContentAuthorizationError('Slack denied access to the private file. Verify the files:read bot scope, app installation, and channel membership.') from exc
            if exc.code == 404:
                raise content_ingestion.ContentError('The shared Slack file no longer exists or is unavailable.') from exc
            if (exc.code == 429 or 500 <= exc.code < 600) and attempt < attempts:
                sleeper(min(2 ** (attempt - 1), 4))
                continue
            if exc.code == 429 or 500 <= exc.code < 600:
                raise content_ingestion.TemporaryContentError('Slack temporarily could not download the shared file.') from exc
            raise content_ingestion.ContentError(f'Slack file download failed with HTTP status {exc.code}.') from exc
        except (_content_ingestion__URLError, TimeoutError) as exc:
            log_exception(content_ingestion.logger, 'Slack file download failed', exc, function='download', file_id=metadata['file_id'], attempt=attempt)
            if attempt < attempts:
                sleeper(min(2 ** (attempt - 1), 4))
                continue
            raise content_ingestion.TemporaryContentError('Slack temporarily could not download the shared file.') from exc
        except Exception as exc:
            log_exception(content_ingestion.logger, 'Slack file download failed', exc, function='download', file_id=metadata['file_id'], attempt=attempt)
            raise content_ingestion.ContentError('Slack did not provide accessible file content.') from exc
    else:
        raise content_ingestion.TemporaryContentError('Slack temporarily could not download the shared file.')
    if len(data) > max_bytes:
        raise content_ingestion.ContentError('The shared file exceeds the configured media size limit.')
    if not data:
        raise content_ingestion.ContentError('The shared Slack file is empty.')
    return data
content_ingestion.download = _content_ingestion__download
def _content_ingestion__normalize_transcript(value):
    text = str(value or '').replace('\x00', '')
    text = _content_ingestion__re.sub('^WEBVTT.*?$', '', text, flags=_content_ingestion__re.I | _content_ingestion__re.M)
    text = _content_ingestion__re.sub('^\\s*\\d+\\s*$', '', text, flags=_content_ingestion__re.M)
    text = _content_ingestion__re.sub('^\\s*\\d{1,2}:\\d{2}(?::\\d{2})?[.,]\\d+\\s+-->.*$', '', text, flags=_content_ingestion__re.M)
    text = _content_ingestion__re.sub('[ \\t]+', ' ', text)
    text = _content_ingestion__re.sub('\\n{3,}', '\n\n', text).strip()
    return text
content_ingestion.normalize_transcript = _content_ingestion__normalize_transcript
def _content_ingestion__transcript_quality_issue(value, *, duration_seconds=None, confidence=None):
    """Return a conservative reason when speech text is unsafe to interpret."""
    text = str(value or '').strip()
    words = _content_ingestion__re.findall('[A-Za-z0-9]+', text)
    try:
        minimum_confidence = float(_content_ingestion__os.getenv('STT_MIN_TRANSCRIPT_CONFIDENCE', '0.25'))
    except ValueError:
        minimum_confidence = 0.25
    if confidence is not None and confidence < minimum_confidence:
        return 'low provider confidence'
    if duration_seconds and duration_seconds >= 2 and (len(words) < 2):
        return 'too little speech was recognized'
    if _content_ingestion__re.search('\\b(?:jupy|g\\s*p|gp)\\s+(?:one|two|three|four)\\b', text, _content_ingestion__re.I):
        return 'a priority token was not recognized reliably'
    if _content_ingestion__re.search('\\b([A-Za-z]+)(?:\\s+\\1){3,}\\b', text, _content_ingestion__re.I):
        return 'the transcript contains repeated recognition artifacts'
    if _content_ingestion__re.fullmatch('\\s*[\\[(]?(?:music|noise|inaudible|silence)[\\])]?\\s*[.!]?\\s*', text, _content_ingestion__re.I):
        return 'no intelligible request was recognized'
    return None
content_ingestion.transcript_quality_issue = _content_ingestion__transcript_quality_issue
def _content_ingestion___pasted_transcript(text):
    lines = str(text or '').strip().splitlines()
    if len(lines) > 1 and content_ingestion.extraction_requested(text):
        content_lines = [line for line in lines if not content_ingestion.extraction_requested(line)]
        return '\n'.join(content_lines).strip()
    marker = _content_ingestion__re.search('\\btranscript\\s*:\\s*', str(text or ''), _content_ingestion__re.I)
    if marker:
        return str(text or '')[marker.end():].strip()
    inline = _content_ingestion__re.match('^.*?\\b(?:action\\s+items?|tasks?|todos?)\\b[^:]*:\\s*(.+)$', str(text or ''), _content_ingestion__re.I | _content_ingestion__re.S)
    return inline.group(1).strip() if inline and content_ingestion.extraction_requested(text) else ''
content_ingestion._pasted_transcript = _content_ingestion___pasted_transcript
def _content_ingestion__ingest(text, files=(), attachments=(), bot_token='', downloader=content_ingestion.download, transcriber=transcription.transcribe_bytes, slack_client=None, metadata_resolver=content_ingestion.resolve_file_metadata):
    sources = list(files or [])
    for attachment in attachments or []:
        if isinstance(attachment, dict) and content_ingestion.file_kind(attachment) != 'unsupported':
            sources.append(attachment)
    content_ingestion.logger.info('shared_content_received files=%d attachments=%d text_chars=%d', len(files or []), len(attachments or []), len(str(text or '')))
    content_ingestion.logger.info('media_received files=%d attachments=%d', len(files or []), len(attachments or []))
    results, errors = ([], [])
    for source in sources:
        display_label = str(source.get('title') or source.get('name') or 'shared file')
        try:
            source = metadata_resolver(source, slack_client)
        except content_ingestion.ContentError as exc:
            errors.append(f'{display_label}: {exc}')
            continue
        kind = content_ingestion.file_kind(source)
        file_id = str(source.get('id') or 'unknown')
        content_ingestion.logger.info('media_type_detected file_id=%s type=%s mimetype=%s filetype=%s', file_id, kind, str(source.get('mimetype') or 'unknown'), str(source.get('filetype') or 'unknown'))
        display_label = str(source.get('title') or source.get('name') or 'shared file')
        if kind == 'unsupported':
            errors.append(f'{display_label}: unsupported content type')
            continue
        try:
            content_ingestion.logger.info('media_download_started file_id=%s source_type=%s', file_id, kind)
            raw = downloader(source, bot_token)
            content_ingestion.logger.info('media_download_completed file_id=%s source_type=%s bytes=%d', file_id, kind, len(raw))
            if kind == 'transcript':
                transcript = content_ingestion.normalize_transcript(raw.decode('utf-8-sig'))
                chunks = 1
            else:
                content_ingestion.logger.info('transcription_started file_id=%s source_type=%s bytes=%d', file_id, kind, len(raw))
                if transcriber is transcription.transcribe_bytes:
                    observed = transcriber(raw, kind, str(source.get('mimetype') or ''), file_id=file_id)
                else:
                    observed = transcriber(raw, kind, str(source.get('mimetype') or ''))
                raw_transcript = str(observed.text or '')
                content_ingestion.logger.info('whisper_transcript_raw source=%s file_id=%s transcript=%r', kind, file_id, raw_transcript)
                transcript, chunks = (content_ingestion.normalize_transcript(raw_transcript), observed.chunks)
                content_ingestion.logger.info('whisper_transcript_normalized source=%s file_id=%s transcript=%r', kind, file_id, transcript)
                quality_issue = content_ingestion.transcript_quality_issue(transcript, duration_seconds=observed.duration_seconds, confidence=getattr(observed, 'confidence', None))
                if quality_issue:
                    content_ingestion.logger.warning('transcription_quality_rejected file_id=%s media_type=%s reason=%s confidence=%s', file_id, kind, quality_issue, getattr(observed, 'confidence', None))
                    raise content_ingestion.ContentError("I couldn't confidently understand this recording. Please repeat it clearly. No task changes were made.")
                content_ingestion.logger.info('transcription_completed file_id=%s media_type=%s chunk_count=%d transcript_chars=%d', file_id, kind, chunks, len(transcript))
            if not transcript:
                raise content_ingestion.ContentError('The transcript is empty or contains no usable text.')
            content_ingestion.logger.info('transcript_normalized source=%s file_id=%s transcript_chars=%d', kind, file_id, len(transcript))
            results.append(content_ingestion.IngestedContent(transcript, kind, file_id, chunks))
        except (content_ingestion.ContentError, transcription.TranscriptionError, UnicodeDecodeError) as exc:
            if kind in {'audio', 'video'}:
                content_ingestion.logger.warning('transcription_failed file_id=%s media_type=%s stage=%s error_type=%s message=%s', file_id, kind, getattr(exc, 'stage', 'transcript_normalization'), type(exc).__name__, redact(exc))
            errors.append(f'{display_label}: {exc}')
    pasted = content_ingestion._pasted_transcript(text)
    if pasted:
        normalized = content_ingestion.normalize_transcript(pasted)
        if normalized:
            results.append(content_ingestion.IngestedContent(normalized, 'transcript', 'Slack message'))
    if not results:
        if errors:
            raise content_ingestion.ContentError('; '.join(errors))
        raise content_ingestion.ContentError('Attach accessible audio/video, or include transcript text to extract action items.')
    return (results, errors)
content_ingestion.ingest = _content_ingestion__ingest


# references.py
'Pure reference grammar. Only complete noun phrases are references, never title substrings.'
import re as _references__re
references.re = _references__re
from dataclasses import dataclass as _references__dataclass
references.dataclass = _references__dataclass
from enum import Enum as _references__Enum
references.Enum = _references__Enum
@_references__dataclass(frozen=True)
class _references__Reference:
    kind: str
    positions: tuple = ()
    count: int = 0
references.Reference = _references__Reference
class _references__TargetType(str, _references__Enum):
    SINGLE_ITEM = 'single_item'
    MULTIPLE_ITEMS = 'multiple_items'
    FILTERED_COLLECTION = 'filtered_collection'
    ALL_APPLICABLE_ITEMS = 'all_applicable_items'
    CONTEXTUAL_ITEMS = 'contextual_items'
references.TargetType = _references__TargetType
class _references__TargetCardinality(str, _references__Enum):
    SINGLE = 'single'
    MULTIPLE = 'multiple'
    COLLECTION = 'collection'
    AMBIGUOUS = 'ambiguous'
references.TargetCardinality = _references__TargetCardinality
@_references__dataclass(frozen=True)
class _references__ResolvedTargetSet:
    target_type: references.TargetType
    item_ids: tuple
    items: tuple
references.ResolvedTargetSet = _references__ResolvedTargetSet
_references___ORDINALS = dict(zip('first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split(), range(1, 21)))
references._ORDINALS = _references___ORDINALS
_references___TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
references._TENS = _references___TENS
_references___CARDINALS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
references._CARDINALS = _references___CARDINALS
references._ORDINALS.update({'thirtieth': 30, 'fortieth': 40, 'fiftieth': 50, 'sixtieth': 60, 'seventieth': 70, 'eightieth': 80, 'ninetieth': 90, 'hundredth': 100})
_references___COLLECTION_NOUN = '(?:(?:action|todo|to-do)\\s+)?(?:tasks?|items?|entries|work)'
references._COLLECTION_NOUN = _references___COLLECTION_NOUN
_references___COLLECTION_QUANTIFIER = '(?:all|every|each|entire|whole)'
references._COLLECTION_QUANTIFIER = _references___COLLECTION_QUANTIFIER
def _references__collection_scope(text):
    """Classify collection grammar independently from any operation wording."""
    value = str(text or '').strip().casefold().rstrip('.!?')
    explicit = _references__re.search(f'\\b{references._COLLECTION_QUANTIFIER}\\b(?:\\s+of)?(?:\\s+(?:the|my|current|available|existing|pending|open|completed|overdue|due|high|low|priority)){{0,5}}\\s+{references._COLLECTION_NOUN}\\b', value)
    if explicit or _references__re.search('\\beverything\\b', value):
        return 'all_applicable'
    if _references__re.search('\\b(?:all|both)\\b(?:\\s+of)?\\s+(?:these|those|them)\\b', value):
        return 'contextual'
    return None
references.collection_scope = _references__collection_scope
def _references__ordinal(text):
    text = text.strip().lower().replace('-', ' ')
    if _references__re.fullmatch('#?\\d+(?:st|nd|rd|th)?', text):
        return int(_references__re.sub('\\D', '', text))
    if text in references._ORDINALS:
        return references._ORDINALS[text]
    words = text.split()
    if len(words) == 2 and words[0] in references._TENS and (words[1] in references._ORDINALS):
        unit = references._ORDINALS[words[1]]
        if unit < 10:
            return references._TENS[words[0]] + unit
    return None
references.ordinal = _references__ordinal
def _references__parse_reference(text):
    text = str(text or '').strip().casefold().rstrip('.!?')
    if any((c in text for c in '"“”‘’')):
        return None
    text = _references__re.sub('^(?:the\\s+)', '', text)
    text = _references__re.sub('\\s+(?:on|from|in)\\s+(?:that|this|the)\\s+(?:displayed\\s+)?list$', '', text)
    text = _references__re.sub('\\s+(?:from\\s+)?above$', '', text)
    scope = references.collection_scope(text)
    if scope:
        return references.Reference('both' if scope == 'contextual' and text.startswith('both') else 'all')
    if _references__re.fullmatch('(?:task|item|one)\\s+(?:that\\s+)?you\\s+(?:just\\s+)?(?:showed|created|updated|completed|mentioned)', text):
        return references.Reference('focus')
    if text in {'its', 'their task'}:
        return references.Reference('focus')
    if text in {'those two', 'these two', 'the two'}:
        return references.Reference('both')
    if text in {'it', 'that', 'this', 'that task', 'this task', 'that item', 'this item', 'that one', 'this one', 'the one above', 'one above', 'the one i mentioned', 'task i mentioned', 'the task i mentioned', 'task we discussed', 'the task we discussed', 'the one we discussed'}:
        return references.Reference('focus')
    if text in {'them', 'these', 'those', 'these tasks', 'those tasks', 'these items', 'those items'}:
        return references.Reference('focus_set')
    if _references__re.fullmatch('(?:all|both)(?:\\s+(?:of\\s+)?(?:them|these|those|the tasks|tasks|items|action items))?', text):
        return references.Reference('both' if text.startswith('both') else 'all')
    if text == 'everything':
        return references.Reference('all')
    if _references__re.fullmatch('(?:previous|prior)(?:\\s+(?:one|task|item|entry))?', text):
        return references.Reference('previous')
    if _references__re.fullmatch('(?:next|following|another)(?:\\s+(?:one|task|item|entry))?', text):
        return references.Reference('relative', (1,))
    if _references__re.fullmatch('(?:(?:most\\s+)?recent|latest|newest|last|final)(?:\\s+(?:one|task|item|entry))?', text):
        return references.Reference('positions', (-1,))
    if _references__re.fullmatch('(?:earliest|oldest)(?:\\s+(?:one|task|item|entry))?', text):
        return references.Reference('positions', (1,))
    count = _references__re.fullmatch('(?:first|last)\\s+(\\d+|one|two|three|four|five|six|seven|eight|nine|ten)(?:\\s+(?:tasks|items|ones))?', text)
    if count:
        amount = int(count[1]) if count[1].isdigit() else references._CARDINALS[count[1]]
        return references.Reference('tail' if text.startswith('last') else 'head', count=amount)
    text = _references__re.sub('^(?:tasks?|items?|entr(?:y|ies)|numbers?|nos?\\.)\\s+', '', text)
    text = _references__re.sub('\\s+(?:ones?|tasks?|items?|entr(?:y|ies))$', '', text)
    text = _references__re.sub('\\bthe\\s+', '', text)
    span = _references__re.fullmatch('(#?\\d+)(?:\\s*(?:-|through|to)\\s*)(#?\\d+)', text)
    if span:
        start, end = (references.ordinal(span[1]), references.ordinal(span[2]))
        return references.Reference('positions', tuple(range(start, end + 1))) if end >= start else references.Reference('positions')
    parts = _references__re.split('\\s*(?:,\\s*(?:and\\s+)?|\\band\\b|&)\\s*', text)
    positions = [-1 if part in {'last', 'final'} else references.ordinal(part) for part in parts]
    if positions and all((p is not None for p in positions)):
        return references.Reference('positions', tuple(dict.fromkeys(positions)))
    return None
references.parse_reference = _references__parse_reference
def _references__extract_contextual_reference(text):
    """Extract reference grammar embedded in a larger request without parsing intent."""
    value = str(text or '').casefold().replace('-', ' ')
    quantity = _references__re.search('\\b(first|last)\\s+(\\d+|one|two|three|four|five|six|seven|eight|nine|ten)\\b', value)
    if quantity:
        amount = int(quantity.group(2)) if quantity.group(2).isdigit() else references._CARDINALS[quantity.group(2)]
        return references.Reference('tail' if quantity.group(1) == 'last' else 'head', count=amount)
    tokens = _references__re.findall('[a-z][a-z0-9]*|#?\\d+(?:st|nd|rd|th)?', value)
    noun_positions = {i for i, token in enumerate(tokens) if token in {'task', 'tasks', 'item', 'items', 'one', 'ones', 'entry', 'entries'}}
    modifiers = {'and', 'or', 'the', 'pending', 'open', 'completed', 'overdue', 'these', 'those', 'displayed'}

    def qualifies(index):
        for noun in sorted((position for position in noun_positions if position > index)):
            between = tokens[index + 1:noun]
            if all((token in modifiers or references.ordinal(token) is not None or token in {'last', 'final'} for token in between)):
                return True
            break
        return False
    positions = []
    for index, token in enumerate(tokens):
        position = references.ordinal(token)
        if position is not None and qualifies(index):
            positions.append(position)
        elif token in {'last', 'final'} and qualifies(index):
            positions.append(-1)
    if positions:
        return references.Reference('positions', tuple(dict.fromkeys(positions)))
    for index, token in enumerate(tokens):
        if token in {'next', 'following', 'another'} and any((index < noun <= index + 2 for noun in noun_positions)):
            return references.Reference('relative', (1,))
        if token in {'previous', 'prior'} and any((index < noun <= index + 2 for noun in noun_positions)):
            return references.Reference('previous')
    return None
references.extract_contextual_reference = _references__extract_contextual_reference
def _references__reference_from(parsed):
    """Accept the normalized grammar plus legacy parser fields at one boundary."""
    if parsed.get('literal_name'):
        return None
    value = parsed.get('reference')
    if isinstance(value, references.Reference):
        return value
    if isinstance(value, dict):
        return references.Reference(value['kind'], tuple(value.get('positions', ())), value.get('count', 0))
    if parsed.get('selection_numbers'):
        return references.Reference('positions', tuple(parsed['selection_numbers']))
    if parsed.get('selection_count'):
        return references.Reference('head', count=int(parsed['selection_count']))
    if parsed.get('selection_index') is not None:
        return references.Reference('positions', (int(parsed['selection_index']),))
    selection = parsed.get('selection') or parsed.get('task_reference')
    if selection in {'__LAST__', 'single'}:
        return references.Reference('focus')
    if selection:
        return references.parse_reference(selection)
    if parsed.get('task_name') == '__LAST__':
        return references.Reference('focus')
    return references.parse_reference(parsed.get('task_name'))
references.reference_from = _references__reference_from
def _references__requested_cardinality(parsed):
    """Return the user's target-count contract independently of candidate scope."""
    selection = parsed.get('target_selection') or {}
    if selection.get('mode') == 'one':
        return references.TargetCardinality.SINGLE
    if selection.get('mode') == 'many':
        return references.TargetCardinality.MULTIPLE
    if selection.get('mode') == 'collection':
        return references.TargetCardinality.COLLECTION
    reference = references.reference_from(parsed)
    if reference:
        if reference.kind in {'focus', 'previous', 'relative'}:
            return references.TargetCardinality.SINGLE
        if reference.kind == 'positions':
            return references.TargetCardinality.SINGLE if len(reference.positions) == 1 else references.TargetCardinality.MULTIPLE
        if reference.kind in {'head', 'tail'} and reference.count == 1:
            return references.TargetCardinality.SINGLE
        if reference.kind in {'both', 'head', 'tail'}:
            return references.TargetCardinality.MULTIPLE
        if reference.kind in {'all', 'focus_set'}:
            return references.TargetCardinality.COLLECTION
    scope = parsed.get('target_scope')
    if scope == 'single' or parsed.get('limit') == 1:
        return references.TargetCardinality.SINGLE
    if scope == 'multiple' or (isinstance(parsed.get('limit'), int) and parsed['limit'] > 1):
        return references.TargetCardinality.MULTIPLE
    if scope in {'filtered', 'all_applicable', 'contextual'}:
        return references.TargetCardinality.COLLECTION
    if parsed.get('task_name'):
        return references.TargetCardinality.SINGLE
    return references.TargetCardinality.AMBIGUOUS
references.requested_cardinality = _references__requested_cardinality
def _references__resolved_target_type(cardinality, source_scope=None):
    if cardinality == references.TargetCardinality.SINGLE:
        return references.TargetType.SINGLE_ITEM
    if cardinality == references.TargetCardinality.MULTIPLE:
        return references.TargetType.MULTIPLE_ITEMS
    if source_scope == 'all_applicable':
        return references.TargetType.ALL_APPLICABLE_ITEMS
    if source_scope == 'filtered':
        return references.TargetType.FILTERED_COLLECTION
    return references.TargetType.CONTEXTUAL_ITEMS
references.resolved_target_type = _references__resolved_target_type
def _references__enforce_cardinality(parsed, item_ids):
    """Fail closed when resolution expands beyond the user's requested shape."""
    cardinality = references.requested_cardinality(parsed)
    count = len(tuple(item_ids))
    if cardinality == references.TargetCardinality.SINGLE and count != 1:
        raise ValueError(f'I resolved {count} tasks for a singular request. Please clarify the exact task; no changes were made.')
    if cardinality == references.TargetCardinality.MULTIPLE and count < 1:
        raise ValueError("I couldn't resolve the requested tasks; no changes were made.")
    if cardinality == references.TargetCardinality.AMBIGUOUS and count != 1:
        raise ValueError('Please clarify whether you mean one task or a collection; no changes were made.')
    return cardinality
references.enforce_cardinality = _references__enforce_cardinality
def _references__select_ids(reference, displayed_ids, focus_ids=()):
    ids = list(displayed_ids)
    focus = list(focus_ids)
    if reference.kind == 'focus':
        focus = list(focus_ids) or ids
        if len(focus) != 1:
            raise ValueError('Which task do you mean? Please use its displayed number or name.')
        return focus
    if reference.kind == 'previous':
        if len(focus) == 1:
            return focus
        if ids:
            return [ids[-1]]
        raise ValueError('Which previous task do you mean?')
    if reference.kind == 'both':
        if len(focus) == 2:
            return focus
        if len(ids) != 2:
            raise ValueError(f'I found {len(ids)} tasks. Which two tasks do you mean?')
        return ids
    if reference.kind == 'focus_set':
        selected = focus or ids
        if not selected:
            raise ValueError('Which tasks do you mean? Please display or select them first.')
        return selected
    if reference.kind == 'all':
        if not ids:
            raise ValueError('There are no tasks in that displayed list.')
        return ids
    if reference.kind in {'head', 'tail'}:
        if not 1 <= reference.count <= len(ids):
            raise ValueError('That count is outside the displayed list.')
        return ids[:reference.count] if reference.kind == 'head' else ids[-reference.count:]
    if reference.kind == 'relative':
        if not ids or len(focus) != 1 or focus[0] not in ids:
            if reference.positions == (-1,) and ids and (not focus):
                return [ids[-1]]
            raise ValueError('Which task should I use as the starting point for that reference?')
        position = ids.index(focus[0]) + reference.positions[0]
        if not 0 <= position < len(ids):
            raise ValueError('There is no task in that relative position in the displayed list.')
        return [ids[position]]
    if reference.kind != 'positions' or not reference.positions:
        raise ValueError('Please specify a valid displayed position.')
    result = []
    for position in reference.positions:
        if not isinstance(position, int) or (position != -1 and (not 1 <= position <= len(ids))) or (not ids):
            raise ValueError('That position is outside the displayed list.')
        result.append(ids[-1] if position == -1 else ids[position - 1])
    return list(dict.fromkeys(result))
references.select_ids = _references__select_ids


def _render_native_task_table(rows, title):
    """Render every task-list view with the same bounded Slack code-block table."""
    import unicodedata
    from html import unescape

    values = list(rows)
    completed = sum(str(row.status or "").casefold() == "completed" for row in values)
    summary = f"{len(values)} tasks · {len(values) - completed} pending · {completed} completed"
    headings = ("#", "Task", "Assignee", "Priority", "Due Date", "Status", "Reviewer Attachments")
    limits = (max(3, len(str(len(values)))), 32, 12, 8, 14, 10, 24)

    def clean(value):
        value = unescape(str(value if value not in (None, "") else "—"))
        value = value.replace("```", "'''").replace("`", "'")
        value = value.replace("\r", " ").replace("\n", " ").replace("<", "‹").replace(">", "›")
        return " ".join(value.split()) or "—"

    def width(value):
        return sum(0 if unicodedata.combining(char) else
                   2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
                   for char in value)

    def fit(value, limit):
        value = clean(value)
        if width(value) <= limit:
            return value
        clipped = ""
        for char in value:
            if width(clipped + char) > limit - 1:
                break
            clipped += char
        return clipped.rstrip() + "…"

    table_rows = []
    link_lines = []
    for position, row in enumerate(values, 1):
        attachments = row.reviewer_attachments or ()
        attachment_label = (attachments[0][0] if len(attachments) == 1 else
                            f"{len(attachments)} files" if attachments else "—")
        for name, url in attachments:
            if url:
                safe_name = slack_presentation.text(clean(name)).replace("|", " ")
                link_lines.append(f"• {position}: <{url}|{safe_name}>")
        table_rows.append((
            str(position), row.name or "Unnamed task",
            row.assignee if row.assignee not in (None, "Unassigned") else "—",
            row.priority if row.priority not in (None, "No priority") else "—",
            row.due_date if row.show_due and row.due_date else "—",
            "Completed" if str(row.status or "").casefold() == "completed" else "Pending",
            attachment_label,
        ))

    prepared = [tuple(fit(value, limit) for value, limit in zip(row, limits))
                for row in (headings, *table_rows)]
    widths = [max(width(row[index]) for row in prepared) for index in range(len(headings))]

    def line(row):
        return "  ".join(value + " " * (column_width - width(value))
                         for value, column_width in zip(row, widths))

    # Slack encodes trailing code-block padding as &#x20; in some clients.
    # Interior padding keeps columns aligned; the last column needs none.
    header = line(prepared[0]).rstrip()
    body_lines = [line(row).rstrip() for row in prepared[1:]]
    separator = "-" * max([width(header), *(width(row) for row in body_lines)])
    body = "\n".join(body_lines)
    table = f"```\n{header}\n{separator}" + (f"\n{body}" if body else "") + "\n```"
    response = f"*{slack_presentation.text(title)}*\n\n{summary}\n\n{table}"
    if link_lines:
        response += "\n\n*Reviewer Attachment Links*\n" + "\n".join(link_lines)
    return response


slack_presentation.native_task_table = _render_native_task_table
