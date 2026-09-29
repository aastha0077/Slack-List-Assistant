"""Best-effort append-only history for verified Slack List mutations."""
import json
import sqlite3
import time
import uuid

import slack_tools


def _connect(db_path):
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mutation_audit (
            audit_id TEXT PRIMARY KEY,
            created REAL NOT NULL,
            team_id TEXT,
            channel_id TEXT,
            thread_ts TEXT,
            actor_id TEXT,
            actor_role TEXT,
            list_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            changes_json TEXT NOT NULL,
            before_json TEXT,
            after_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_tracking (
            list_id TEXT PRIMARY KEY,
            started REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def field_snapshot(item, schema):
    if item is None:
        return None
    return {
        "id": slack_tools.extract_item_id(item),
        "name": slack_tools.extract_item_name(item, schema),
        "assignees": slack_tools.extract_assignee_ids(item, schema),
        "due_date": slack_tools.extract_due_date(item, schema),
        "priority": slack_tools.extract_priority(item, schema),
        "completed": slack_tools.extract_completed(item, schema),
        "fields": item.get("fields") or item.get("cells") or [],
    }


def record(db_path, ctx, item_id, operation, changes, schema, before=None, after=None):
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT MIN(created) FROM mutation_audit WHERE list_id=?", (ctx.list_id,)).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO audit_tracking VALUES (?, ?)",
            (ctx.list_id, existing or time.time()),
        )
        conn.execute(
            "INSERT INTO mutation_audit VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), time.time(), ctx.team_id, ctx.channel_id, ctx.thread_ts,
             ctx.user_id, ctx.role, ctx.list_id, item_id, operation,
             json.dumps(changes, default=str),
             json.dumps(field_snapshot(before, schema), default=str) if before is not None else None,
             json.dumps(field_snapshot(after, schema), default=str) if after is not None else None))


def history(db_path, list_id, item_ids=(), limit=100, since=None, until=None):
    query = "SELECT created, actor_id, actor_role, item_id, operation, changes_json, before_json, after_json FROM mutation_audit WHERE list_id=?"
    params = [list_id]
    if item_ids:
        placeholders = ",".join("?" for _ in item_ids)
        query += f" AND item_id IN ({placeholders})"
        params.extend(item_ids)
    if since is not None:
        query += " AND created>=?"
        params.append(float(since))
    if until is not None:
        query += " AND created<?"
        params.append(float(until))
    query += " ORDER BY created DESC LIMIT ?"
    params.append(limit)
    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {"created": row[0], "actor_id": row[1], "actor_role": row[2], "item_id": row[3],
         "operation": row[4], "changes": json.loads(row[5]),
         "before": json.loads(row[6]) if row[6] else None,
         "after": json.loads(row[7]) if row[7] else None}
        for row in rows
    ]


def tracking_started(db_path, list_id):
    """Return the truthful start of locally available history for one List."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT started FROM audit_tracking WHERE list_id=?", (list_id,)).fetchone()
        if row:
            return row[0]
        earliest = conn.execute(
            "SELECT MIN(created) FROM mutation_audit WHERE list_id=?", (list_id,)).fetchone()[0]
        started = earliest or time.time()
        conn.execute("INSERT OR IGNORE INTO audit_tracking VALUES (?, ?)", (list_id, started))
        return started
