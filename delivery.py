"""Durable event responses and operation checkpoints for Slack redelivery.

SQLite records intent before writes. A retry resumes exact IDs; a create whose
result is uncertain is reconciled, never blindly repeated. Socket Mode remains
the transport. This module does not create or configure a Slack integration.
"""
import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar

_active = ContextVar("slack_event", default=None)


def _connect(db):
    conn = db()
    conn.execute("CREATE TABLE IF NOT EXISTS delivery (key TEXT PRIMARY KEY, status TEXT NOT NULL, updated REAL NOT NULL, data TEXT NOT NULL)")
    conn.commit()
    return conn


def read(db, key):
    with _connect(db) as conn:
        row = conn.execute("SELECT status, updated, data FROM delivery WHERE key=?", (key,)).fetchone()
    conn.close()
    return {"status": row[0], "updated": row[1], **json.loads(row[2])} if row else None


def write(db, key, status, data):
    with _connect(db) as conn:
        conn.execute("INSERT OR REPLACE INTO delivery VALUES (?, ?, ?, ?)",
                     (key, status, time.time(), json.dumps(data)))
    conn.close()


def claim(db, key):
    conn = _connect(db)
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
                     (key, "processing", time.time(), json.dumps(data)))
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
    digest = hashlib.sha256(json.dumps([kind, identity], sort_keys=True).encode()).hexdigest()
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
