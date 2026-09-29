"""Deterministic predictive intelligence over an authorized task snapshot.

This module never reads Slack directly and never mutates task state.  It models
pressure from facts already present in ``NormalizedTask`` records and uses
careful language: a signal is an emerging risk, not a prediction of failure.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Iterable

from project_intelligence import NormalizedTask


@dataclass(frozen=True)
class DeadlineCluster:
    due_date: date
    task_ids: tuple[str, ...]
    priority_counts: dict[str, int]
    owner_counts: dict[str | None, int]


@dataclass(frozen=True)
class WorkloadOutlook:
    owner_id: str | None
    pending: int
    overdue: int
    p1: int
    due_within_48h: int
    due_within_7d: int


@dataclass(frozen=True)
class EmergingRisk:
    task_id: str
    task_name: str
    owner_ids: tuple[str, ...]
    priority: str | None
    due_date: date | None
    level: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class PredictiveSummary:
    pending: int
    completed: int
    overdue: int
    due_today: int
    due_within_24h: int
    due_within_48h: int
    due_within_7d: int
    priority_counts: dict[str, int]
    unassigned: int
    workload: tuple[WorkloadOutlook, ...]
    deadline_clusters: tuple[DeadlineCluster, ...]
    emerging_risks: tuple[EmergingRisk, ...]


def _pending(tasks: Iterable[NormalizedTask]) -> list[NormalizedTask]:
    return [task for task in tasks if not task.completed]


def calculate_deadline_pressure(tasks: Iterable[NormalizedTask], today: date) -> dict[str, int]:
    """Return mutually understandable deadline facts for pending work."""
    pending = _pending(tasks)
    return {
        "overdue": sum(bool(task.due_date and task.due_date < today) for task in pending),
        "due_today": sum(task.due_date == today for task in pending),
        "due_within_24h": sum(bool(task.due_date and today <= task.due_date <= today + timedelta(days=1))
                              for task in pending),
        "due_within_48h": sum(bool(task.due_date and today <= task.due_date <= today + timedelta(days=2))
                              for task in pending),
        "due_within_7d": sum(bool(task.due_date and today <= task.due_date <= today + timedelta(days=7))
                             for task in pending),
    }


def calculate_priority_pressure(tasks: Iterable[NormalizedTask]) -> dict[str, int]:
    pending = _pending(tasks)
    return {priority: sum(task.priority == priority for task in pending)
            for priority in ("P1", "P2", "P3")}


def calculate_workload_pressure(tasks: Iterable[NormalizedTask], today: date) -> tuple[WorkloadOutlook, ...]:
    """Calculate current and near-term owner load without subjective labels."""
    grouped: dict[str | None, list[NormalizedTask]] = defaultdict(list)
    for task in _pending(tasks):
        if task.owner_ids:
            for owner_id in task.owner_ids:
                grouped[owner_id].append(task)
        else:
            grouped[None].append(task)
    rows = []
    for owner_id, owned in grouped.items():
        rows.append(WorkloadOutlook(
            owner_id=owner_id,
            pending=len(owned),
            overdue=sum(bool(task.due_date and task.due_date < today) for task in owned),
            p1=sum(task.priority == "P1" for task in owned),
            due_within_48h=sum(bool(task.due_date and today <= task.due_date <= today + timedelta(days=2))
                               for task in owned),
            due_within_7d=sum(bool(task.due_date and today <= task.due_date <= today + timedelta(days=7))
                              for task in owned),
        ))
    return tuple(sorted(rows, key=lambda row: (-row.pending, row.owner_id or "")))


def calculate_deadline_concentration(
        tasks: Iterable[NormalizedTask], today: date,
        *, minimum_tasks: int = 2, horizon_days: int = 7) -> tuple[DeadlineCluster, ...]:
    """Find near-term dates shared by multiple pending tasks."""
    grouped: dict[date, list[NormalizedTask]] = defaultdict(list)
    horizon = today + timedelta(days=horizon_days)
    for task in _pending(tasks):
        if task.due_date and today <= task.due_date <= horizon:
            grouped[task.due_date].append(task)
    clusters = []
    for due_date, due_tasks in grouped.items():
        if len(due_tasks) < minimum_tasks:
            continue
        owners: Counter[str | None] = Counter()
        priorities: Counter[str] = Counter()
        for task in due_tasks:
            priorities[task.priority or "No priority"] += 1
            if task.owner_ids:
                owners.update(task.owner_ids)
            else:
                owners[None] += 1
        clusters.append(DeadlineCluster(
            due_date, tuple(sorted(task.item_id for task in due_tasks)),
            dict(priorities), dict(owners)))
    return tuple(sorted(clusters, key=lambda cluster: cluster.due_date))


def build_risk_evidence(task: NormalizedTask, *, today: date,
                        owner_outlook: WorkloadOutlook | None,
                        cluster: DeadlineCluster | None) -> tuple[str, ...]:
    """Build factual evidence for one not-yet-overdue task."""
    evidence = []
    if task.due_date:
        days = (task.due_date - today).days
        if days == 0:
            evidence.append("Due today")
        elif days == 1:
            evidence.append("1 day until deadline")
        elif 1 < days <= 7:
            evidence.append(f"{days} days until deadline")
    if task.priority == "P1":
        evidence.append("P1 priority")
    if not task.owner_ids:
        evidence.append("No assigned owner")
    if owner_outlook and owner_outlook.p1 >= 2:
        evidence.append(f"Owner has {owner_outlook.p1} pending P1 tasks")
    if owner_outlook and owner_outlook.due_within_48h >= 2:
        evidence.append(f"Owner has {owner_outlook.due_within_48h} deadlines within 48 hours")
    if cluster:
        evidence.append(f"{len(cluster.task_ids)} tasks share this deadline")
    return tuple(evidence)


def detect_emerging_risks(tasks: Iterable[NormalizedTask], today: date) -> tuple[EmergingRisk, ...]:
    """Identify evidence-backed pressure before tasks become overdue."""
    values = list(tasks)
    workloads = {row.owner_id: row for row in calculate_workload_pressure(values, today)}
    clusters = calculate_deadline_concentration(values, today)
    cluster_by_task = {task_id: cluster for cluster in clusters for task_id in cluster.task_ids}
    risks = []
    for task in _pending(values):
        if not task.due_date or task.due_date < today or task.due_date > today + timedelta(days=7):
            continue
        owner = workloads.get(task.owner_ids[0]) if len(task.owner_ids) == 1 else None
        evidence = build_risk_evidence(
            task, today=today, owner_outlook=owner, cluster=cluster_by_task.get(task.item_id))
        days = (task.due_date - today).days
        strong_signals = sum((task.priority == "P1", not task.owner_ids,
                              bool(owner and owner.p1 >= 2), task.item_id in cluster_by_task))
        qualifies = days <= 2 and strong_signals >= 1 or days <= 7 and strong_signals >= 2
        if not qualifies:
            continue
        level = "high" if days <= 1 and (task.priority == "P1" or not task.owner_ids) else "attention"
        risks.append(EmergingRisk(
            task.item_id, task.name, task.owner_ids, task.priority,
            task.due_date, level, evidence))
    return tuple(sorted(
        risks, key=lambda risk: (risk.due_date or date.max,
                                 0 if risk.priority == "P1" else 1,
                                 risk.task_name.casefold())))


def build_predictive_summary(tasks: Iterable[NormalizedTask], today: date) -> PredictiveSummary:
    """Build one reusable summary from a single authorized snapshot."""
    values = list(tasks)
    pending = _pending(values)
    deadline = calculate_deadline_pressure(values, today)
    return PredictiveSummary(
        pending=len(pending), completed=len(values) - len(pending),
        overdue=deadline["overdue"], due_today=deadline["due_today"],
        due_within_24h=deadline["due_within_24h"],
        due_within_48h=deadline["due_within_48h"], due_within_7d=deadline["due_within_7d"],
        priority_counts=calculate_priority_pressure(values),
        unassigned=sum(not task.owner_ids for task in pending),
        workload=calculate_workload_pressure(values, today),
        deadline_clusters=calculate_deadline_concentration(values, today),
        emerging_risks=detect_emerging_risks(values, today),
    )


def workload_forecast(summary: PredictiveSummary,
                      name_for_user: Callable[[str], str]) -> tuple[str, ...]:
    """Render factual workload outlook lines from currently known tasks only."""
    lines = []
    for row in summary.workload:
        name = name_for_user(row.owner_id) if row.owner_id else "Unassigned"
        lines.append(
            f"{name} · {row.pending} pending · {row.due_within_48h} due <48h · "
            f"{row.due_within_7d} due in 7 days")
    return tuple(lines)
