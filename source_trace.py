"""Persistent source evidence for tasks created from shared content."""
import json
import sqlite3
import time


def _connect(db_path):
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS task_sources (
            list_id TEXT NOT NULL, item_id TEXT NOT NULL, created REAL NOT NULL,
            source_type TEXT NOT NULL, source_reference TEXT,
            confidence REAL, evidence TEXT, context_json TEXT NOT NULL,
            PRIMARY KEY (list_id, item_id)
        )
    """)
    conn.commit()
    return conn


def record(db_path, ctx, item_id, source):
    source = dict(source or {})
    context = {"team_id": ctx.team_id, "channel_id": ctx.channel_id,
               "thread_ts": ctx.thread_ts, "actor_id": ctx.user_id}
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO task_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ctx.list_id, item_id, time.time(), source.get("type", "text"),
             source.get("reference"), source.get("confidence"),
             str(source.get("evidence") or "")[:240], json.dumps(context)))


def get(db_path, list_id, item_id):
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT created, source_type, source_reference, confidence, evidence, context_json "
            "FROM task_sources WHERE list_id=? AND item_id=?", (list_id, item_id)).fetchone()
    if not row:
        return None
    return {"created": row[0], "source_type": row[1], "source_reference": row[2],
            "confidence": row[3], "evidence": row[4], "context": json.loads(row[5])}
