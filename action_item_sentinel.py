"""Deterministic, persistent monitoring over normalized Slack List tasks."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import project_intelligence
import predictive_intelligence


logger = logging.getLogger("slack_list.sentinel")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


@dataclass(frozen=True)
class SentinelSettings:
    enabled: bool = True
    warning_days: int = 2
    combined_task_threshold: int = 2
    max_alerts_per_scan: int = 100
    approval_ttl_minutes: int = 30

    @classmethod
    def from_env(cls):
        return cls(
            enabled=_env_bool("ACTION_ITEM_SENTINEL_ENABLED", True),
            warning_days=_env_int("SENTINEL_WARNING_DAYS", 2, 1, 14),
            combined_task_threshold=_env_int("SENTINEL_COMBINED_TASK_THRESHOLD", 2, 2, 20),
            max_alerts_per_scan=_env_int("SENTINEL_MAX_ALERTS_PER_SCAN", 100, 1, 1000),
            approval_ttl_minutes=_env_int("SENTINEL_APPROVAL_TTL_MINUTES", 30, 1, 1440),
        )


@dataclass(frozen=True)
class TaskChange:
    task_id: str
    task_name: str
    change_type: str
    previous_value: object
    current_value: object
    timestamp: float


@dataclass(frozen=True)
class SentinelRisk:
    task_id: str
    task_ids: tuple[str, ...]
    task_name: str
    risk_type: str
    severity: str
    reasons: tuple[str, ...]
    owner_ids: tuple[str, ...]
    priority: str | None
    due_date: date | None
    recommendation: str


@dataclass(frozen=True)
class SentinelAlert:
    alert_id: str
    list_id: str
    task_id: str
    event_type: str
    state_hash: str
    severity: str
    payload: dict
    status: str
    created_at: float


@dataclass(frozen=True)
class SentinelEvaluation:
    changes: tuple[TaskChange, ...]
    risks: tuple[SentinelRisk, ...]
    emitted: tuple[SentinelAlert, ...]
    suppressed: int


def task_state(task: project_intelligence.NormalizedTask) -> dict:
    return {
        "task_id": task.item_id, "name": task.name,
        "owner_ids": list(task.owner_ids), "priority": task.priority,
        "due_date": task.due_date.isoformat() if task.due_date else None,
        "completed": task.completed, "status": task.status,
    }


def task_state_hash(task: project_intelligence.NormalizedTask) -> str:
    return hashlib.sha256(json.dumps(task_state(task), sort_keys=True).encode()).hexdigest()


def detect_changes(previous: dict[str, dict], current: Iterable[project_intelligence.NormalizedTask],
                   timestamp: float | None = None) -> list[TaskChange]:
    """Compare reliable snapshots and return only meaningful transitions."""
    timestamp = time.time() if timestamp is None else timestamp
    current_by_id = {task.item_id: task for task in current}
    changes = []
    for task_id, task in current_by_id.items():
        before = previous.get(task_id)
        after = task_state(task)
        if before is None:
            changes.append(TaskChange(task_id, task.name, "created", None, after, timestamp))
            continue
        comparisons = (
            ("completed" if after["completed"] else "reopened", "completed"),
            ("priority_changed", "priority"),
            ("due_date_changed", "due_date"),
            ("owner_changed", "owner_ids"),
        )
        for change_type, field in comparisons:
            if before.get(field) == after.get(field):
                continue
            if field == "priority" and after.get(field) == "P1":
                change_type = "priority_escalated"
            elif field == "due_date":
                earlier, later = before.get(field), after.get(field)
                change_type = "deadline_moved_earlier" if earlier and later and later < earlier else "deadline_moved_later"
            elif field == "owner_ids":
                if not before.get(field) and after.get(field):
                    change_type = "assigned"
                elif before.get(field) and not after.get(field):
                    change_type = "unassigned"
            changes.append(TaskChange(task_id, task.name, change_type,
                                      before.get(field), after.get(field), timestamp))
    for task_id, before in previous.items():
        if task_id not in current_by_id:
            changes.append(TaskChange(task_id, before.get("name") or "Action item",
                                      "deleted", before, None, timestamp))
    return changes


def detect_risks(snapshot: Iterable[project_intelligence.NormalizedTask], today: date,
                 warning_days: int = 2, combined_task_threshold: int = 2) -> list[SentinelRisk]:
    """Detect explainable deadline and workload risks from one snapshot."""
    tasks = [task for task in snapshot if not task.completed]
    # Reuse the shared Task Intelligence detector as the factual candidate set.
    base_ids = {record.item_id for record in project_intelligence.analyze_task_risks(tasks, {}, today)}
    risks = []
    for task in tasks:
        due_delta = (task.due_date - today).days if task.due_date else None
        reasons = []
        risk_type = None
        severity = "attention"
        recommendation = "Confirm completion status or update the deadline."
        if due_delta is not None and due_delta < 0:
            risk_type, severity = "overdue", "critical"
            reasons.extend((f"Overdue by {-due_delta} day{'s' if due_delta != -1 else ''}", "Still pending"))
            if task.priority == "P1":
                reasons.insert(0, "P1 priority")
        elif (not task.owner_ids and task.priority == "P1" and due_delta is not None
              and due_delta <= warning_days):
            risk_type = "unassigned_deadline_risk"
            reasons.extend(("P1 priority", f"Deadline within {warning_days * 24} hours",
                            "No assigned owner", "Still pending"))
            recommendation = "Assign an owner or update the deadline."
        elif task.priority == "P1" and due_delta is not None and due_delta <= 1:
            risk_type = "deadline_risk"
            reasons.extend(("P1 priority", "Deadline within 24 hours", "Still pending"))
        if risk_type and (task.item_id in base_ids or risk_type == "unassigned_deadline_risk"):
            risks.append(SentinelRisk(
                task.item_id, (task.item_id,), task.name, risk_type, severity,
                tuple(reasons), task.owner_ids, task.priority, task.due_date, recommendation))

    urgent_by_owner = {}
    for task in tasks:
        delta = (task.due_date - today).days if task.due_date else None
        if task.priority == "P1" and delta is not None and delta <= warning_days:
            for owner_id in task.owner_ids:
                urgent_by_owner.setdefault(owner_id, []).append(task)
    for owner_id, urgent in urgent_by_owner.items():
        if len(urgent) < combined_task_threshold:
            continue
        risks.append(SentinelRisk(
            f"owner:{owner_id}", tuple(task.item_id for task in urgent),
            "Multiple urgent action items", "combined_workload_risk", "attention",
            (f"{len(urgent)} P1 tasks assigned to the same owner",
             f"Deadlines fall within {warning_days * 24} hours"),
            (owner_id,), "P1", min(task.due_date for task in urgent if task.due_date),
            "Review priorities, ownership, and delivery dates."))
    # Predictive signals are task-specific, deterministic and non-executable.
    # Do not duplicate a task already represented by a current overdue/deadline
    # risk; the predictive signal is for pressure that has not yet materialized.
    represented = {task_id for risk in risks
                   if risk.risk_type != "combined_workload_risk"
                   for task_id in risk.task_ids}
    for emerging in predictive_intelligence.detect_emerging_risks(tasks, today):
        if emerging.task_id in represented:
            continue
        risks.append(SentinelRisk(
            emerging.task_id, (emerging.task_id,), emerging.task_name,
            "emerging_predictive_risk", "attention", emerging.evidence,
            emerging.owner_ids, emerging.priority, emerging.due_date,
            "Review the task before deadline pressure increases."))
    return risks


class SentinelStore:
    """Persistent snapshots, deduplicated alerts, and approval audit."""
    def __init__(self, path: str):
        self.path = str(Path(path))
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sentinel_snapshots (
                    list_id TEXT NOT NULL, task_id TEXT NOT NULL, state_json TEXT NOT NULL,
                    seen_at REAL NOT NULL, PRIMARY KEY (list_id, task_id));
                CREATE TABLE IF NOT EXISTS sentinel_alerts (
                    alert_id TEXT PRIMARY KEY, list_id TEXT NOT NULL, task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL, state_hash TEXT NOT NULL, severity TEXT NOT NULL,
                    payload_json TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL,
                    acted_at REAL, actor_id TEXT);
                CREATE TABLE IF NOT EXISTS sentinel_displays (
                    list_id TEXT NOT NULL, channel_id TEXT NOT NULL, user_id TEXT NOT NULL,
                    position INTEGER NOT NULL, alert_id TEXT NOT NULL, displayed_at REAL NOT NULL,
                    PRIMARY KEY (list_id, channel_id, user_id, position));
            """)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def previous(self, list_id: str) -> dict[str, dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT task_id, state_json FROM sentinel_snapshots WHERE list_id=?", (list_id,)).fetchall()
        return {task_id: json.loads(value) for task_id, value in rows}

    def save_snapshot(self, list_id: str, snapshot, seen_at: float):
        current = {task.item_id: task_state(task) for task in snapshot}
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM sentinel_snapshots WHERE list_id=?", (list_id,))
            connection.executemany(
                "INSERT INTO sentinel_snapshots VALUES (?, ?, ?, ?)",
                [(list_id, task_id, json.dumps(state, sort_keys=True), seen_at)
                 for task_id, state in current.items()])

    def add_alert(self, list_id: str, task_id: str, event_type: str, state_hash: str,
                  severity: str, payload: dict, created_at: float) -> tuple[SentinelAlert, bool]:
        alert_id = hashlib.sha256(
            f"{list_id}|{task_id}|{event_type}|{state_hash}".encode()).hexdigest()
        with self._lock, self._connect() as connection:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO sentinel_alerts VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, NULL, NULL)",
                (alert_id, list_id, task_id, event_type, state_hash, severity,
                 json.dumps(payload, sort_keys=True), created_at)).rowcount == 1
            row = connection.execute(
                "SELECT status, created_at FROM sentinel_alerts WHERE alert_id=?", (alert_id,)).fetchone()
        return SentinelAlert(alert_id, list_id, task_id, event_type, state_hash, severity,
                             payload, row[0], row[1]), inserted

    def resolve_inactive_risks(self, list_id: str, active_ids: set[str], now: float):
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT alert_id FROM sentinel_alerts WHERE list_id=? AND status='active' "
                "AND event_type LIKE 'risk:%'", (list_id,)).fetchall()
            for (alert_id,) in rows:
                if alert_id not in active_ids:
                    connection.execute(
                        "UPDATE sentinel_alerts SET status='resolved', acted_at=? WHERE alert_id=?",
                        (now, alert_id))

    def alerts(self, list_id: str, task_ids=(), status="active", since=None, limit=100):
        query = ("SELECT alert_id, task_id, event_type, state_hash, severity, payload_json, "
                 "status, created_at FROM sentinel_alerts WHERE list_id=?")
        params = [list_id]
        if status:
            query += " AND status=?"
            params.append(status)
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            query += f" AND task_id IN ({placeholders})"
            params.extend(task_ids)
        if since is not None:
            query += " AND created_at>=?"
            params.append(float(since))
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [SentinelAlert(row[0], list_id, row[1], row[2], row[3], row[4],
                              json.loads(row[5]), row[6], row[7]) for row in rows]

    def alert(self, alert_id: str) -> SentinelAlert | None:
        """Load one alert regardless of status for idempotent action responses."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT list_id, task_id, event_type, state_hash, severity, payload_json, "
                "status, created_at FROM sentinel_alerts WHERE alert_id=?", (alert_id,)).fetchone()
        if not row:
            return None
        return SentinelAlert(alert_id, row[0], row[1], row[2], row[3], row[4],
                             json.loads(row[5]), row[6], row[7])

    def save_display(self, list_id: str, channel_id: str, user_id: str,
                     alert_ids: Iterable[str], displayed_at: float):
        """Persist the latest numbered alert view for one authenticated viewer."""
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM sentinel_displays WHERE list_id=? AND channel_id=? AND user_id=?",
                (list_id, channel_id, user_id))
            connection.executemany(
                "INSERT INTO sentinel_displays VALUES (?, ?, ?, ?, ?, ?)",
                [(list_id, channel_id, user_id, position, alert_id, displayed_at)
                 for position, alert_id in enumerate(alert_ids, 1)])

    def resolve_display(self, list_id: str, channel_id: str, user_id: str,
                        position: int) -> tuple[str, float] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT alert_id, displayed_at FROM sentinel_displays "
                "WHERE list_id=? AND channel_id=? AND user_id=? AND position=?",
                (list_id, channel_id, user_id, position)).fetchone()
        return (row[0], row[1]) if row else None

    def claim_action(self, alert_id: str, actor_id: str, expected_hash: str) -> bool:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, state_hash FROM sentinel_alerts WHERE alert_id=?", (alert_id,)).fetchone()
            if not row or row[0] != "active" or row[1] != expected_hash:
                return False
            connection.execute(
                "UPDATE sentinel_alerts SET status='sending', acted_at=?, actor_id=? WHERE alert_id=?",
                (time.time(), actor_id, alert_id))
        return True

    def finish_action(self, alert_id: str, status: str):
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE sentinel_alerts SET status=?, acted_at=? WHERE alert_id=?",
                               (status, time.time(), alert_id))

    def dismiss(self, alert_id: str, actor_id: str) -> bool:
        with self._lock, self._connect() as connection:
            updated = connection.execute(
                "UPDATE sentinel_alerts SET status='dismissed', acted_at=?, actor_id=? "
                "WHERE alert_id=? AND status='active'", (time.time(), actor_id, alert_id)).rowcount
        return bool(updated)


class ActionItemSentinel:
    def __init__(self, store: SentinelStore, settings: SentinelSettings | None = None):
        self.store = store
        self.settings = settings or SentinelSettings.from_env()

    def evaluate(self, list_id: str, snapshot, now: datetime,
                 persist_snapshot: bool = True) -> SentinelEvaluation:
        if not self.settings.enabled:
            return SentinelEvaluation((), (), (), 0)
        snapshot = list(snapshot)
        logger.info("sentinel_evaluation_started list_id=%s task_count=%d llm_used=false", list_id, len(snapshot))
        previous = self.store.previous(list_id) if persist_snapshot else {}
        changes = (detect_changes(previous, snapshot, now.timestamp())
                   if persist_snapshot and previous else [])
        risks = detect_risks(snapshot, now.date(), self.settings.warning_days,
                             self.settings.combined_task_threshold)
        emitted, suppressed, active_risk_ids = [], 0, set()
        events = []
        for change in changes:
            current = next((task for task in snapshot if task.item_id == change.task_id), None)
            state_hash = task_state_hash(current) if current else hashlib.sha256(
                json.dumps(change.previous_value, sort_keys=True).encode()).hexdigest()
            events.append((change.task_id, f"change:{change.change_type}", state_hash,
                           "attention", asdict(change)))
        for risk in risks:
            risk_tasks = [task for task in snapshot if task.item_id in risk.task_ids]
            state = {"risk": risk.risk_type, "tasks": [
                task_state(task) for task in risk_tasks]}
            state_hash = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
            events.append((risk.task_id, f"risk:{risk.risk_type}", state_hash,
                           risk.severity, {**asdict(risk),
                                           "task_state_hash": (task_state_hash(risk_tasks[0])
                                                               if len(risk_tasks) == 1 else None),
                                           "due_date": risk.due_date.isoformat() if risk.due_date else None}))
        for task_id, event_type, state_hash, severity, payload in events[:self.settings.max_alerts_per_scan]:
            alert, inserted = self.store.add_alert(
                list_id, task_id, event_type, state_hash, severity, payload, now.timestamp())
            if event_type.startswith("risk:"):
                active_risk_ids.add(alert.alert_id)
            if inserted:
                emitted.append(alert)
            else:
                suppressed += 1
                logger.info("sentinel_alert_suppressed alert_id=%s reason=duplicate", alert.alert_id)
        if persist_snapshot:
            self.store.resolve_inactive_risks(list_id, active_risk_ids, now.timestamp())
            self.store.save_snapshot(list_id, snapshot, now.timestamp())
        logger.info("sentinel_changes_detected count=%d sentinel_risks_detected count=%d", len(changes), len(risks))
        logger.info("sentinel_alerts_emitted count=%d suppressed=%d llm_used=false llm_call_count=0",
                    len(emitted), suppressed)
        return SentinelEvaluation(tuple(changes), tuple(risks), tuple(emitted), suppressed)
