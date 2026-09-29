"""Persistent, idempotent decision records for task simulations."""
from __future__ import annotations

import json
import hashlib
import sqlite3
import time
from dataclasses import replace

from task_simulation import SimulationResult


class DecisionLedger:
    def __init__(self, path: str):
        self.path = path
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS decision_ledger (
                    decision_id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    requester_id TEXT NOT NULL,
                    list_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    scenario_json TEXT NOT NULL,
                    approval_status TEXT NOT NULL,
                    execution_status TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    actual_json TEXT,
                    plan_id TEXT
                )
            """)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def create(self, result: SimulationResult, list_id: str) -> SimulationResult:
        scoped_fingerprint = hashlib.sha256(
            f"{list_id}|{result.fingerprint}".encode()).hexdigest()
        decision_id = scoped_fingerprint[:12]
        frozen = replace(result, decision_id=decision_id)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO decision_ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (decision_id, result.scenario_id, scoped_fingerprint, result.requester_id,
                 list_id, result.created_at, json.dumps(frozen.to_dict(), default=str),
                 "not_requested", "not_started", "not_verified", None, None))
            row = connection.execute(
                "SELECT scenario_json FROM decision_ledger WHERE fingerprint=?",
                (scoped_fingerprint,)).fetchone()
        return self._result(json.loads(row[0]))

    @staticmethod
    def _result(value: dict) -> SimulationResult:
        for name in ("source_task_ids", "positive", "tradeoffs", "unchanged", "assumptions"):
            value[name] = tuple(value.get(name) or ())
        return SimulationResult(**value)

    def get(self, decision_id: str, requester_id: str, list_id: str):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT scenario_json, approval_status, execution_status, verification_status, "
                "actual_json, plan_id FROM decision_ledger WHERE decision_id=? AND requester_id=? AND list_id=?",
                (decision_id, requester_id, list_id)).fetchone()
        if not row:
            return None
        return {"scenario": self._result(json.loads(row[0])), "approval_status": row[1],
                "execution_status": row[2], "verification_status": row[3],
                "actual": json.loads(row[4]) if row[4] else None, "plan_id": row[5]}

    def get_scenario(self, scenario_id: str, requester_id: str, list_id: str):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT decision_id FROM decision_ledger WHERE scenario_id=? AND requester_id=? "
                "AND list_id=? ORDER BY created_at DESC LIMIT 1",
                (scenario_id, requester_id, list_id)).fetchone()
        return self.get(row[0], requester_id, list_id) if row else None

    def recent(self, requester_id: str, list_id: str, limit: int = 10, failed_only: bool = False):
        where = " AND (execution_status='failed' OR verification_status='variance')" if failed_only else ""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT decision_id, scenario_id, created_at, scenario_json, execution_status, "
                f"verification_status FROM decision_ledger WHERE requester_id=? AND list_id=?{where} "
                "ORDER BY created_at DESC LIMIT ?", (requester_id, list_id, limit)).fetchall()
        return [{"decision_id": row[0], "scenario_id": row[1], "created_at": row[2],
                 "scenario": self._result(json.loads(row[3])),
                 "execution_status": row[4], "verification_status": row[5]} for row in rows]

    def mark_prepared(self, decision_id: str, requester_id: str, list_id: str, plan_id: str):
        with self._connect() as connection:
            connection.execute(
                "UPDATE decision_ledger SET approval_status='required', plan_id=? "
                "WHERE decision_id=? AND requester_id=? AND list_id=?",
                (plan_id, decision_id, requester_id, list_id))

    def record_outcome(self, decision_id: str, requester_id: str, list_id: str,
                       actual: dict, verified: bool):
        with self._connect() as connection:
            connection.execute(
                "UPDATE decision_ledger SET approval_status='approved', execution_status='executed', "
                "verification_status=?, actual_json=? WHERE decision_id=? AND requester_id=? AND list_id=?",
                ("verified" if verified else "variance", json.dumps(actual, default=str),
                 decision_id, requester_id, list_id))
