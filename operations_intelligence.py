"""Read-only, explainable operations intelligence over authorized task data."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re

import predictive_intelligence


MODES = {
    "workload", "risk", "health", "heatmap", "bottlenecks", "briefing",
    "capacity", "executive", "meeting", "unassigned", "collisions",
}


@dataclass(frozen=True)
class TaskRisk:
    task: object
    level: str
    points: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Bottleneck:
    title: str
    detail: str
    severity: str


def parse_request(text):
    """Recognize explicit analytical questions without invoking an LLM."""
    value = re.sub(r"\s+", " ", str(text or "")).strip().rstrip(".?!").casefold()
    workstream = re.fullmatch(
        r"show (testing|deployment|documentation|client|reporting|regression) work", value)
    if workstream:
        theme = workstream.group(1)
        return {"intent": "list", "query": theme, "query_terms": [theme],
                "search": True, "completed": False,
                "result_operation": "return_collection"}
    patterns = (
        (r"(?:give me )?(?:my |today'?s? )?daily briefing", "briefing"),
        (r"(?:give me )?(?:an? )?executive summary", "executive"),
        (r"(?:give me )?(?:a )?team status report", "executive"),
        (r"who (?:is|looks) overloaded", "workload"),
        (r"who has the most overdue work", "workload"),
        (r"who has the most p1 (?:work|tasks)", "workload"),
        (r"(?:show )?(?:team )?(?:capacity|workload intelligence)", "capacity"),
        (r"what are (?:our|the) biggest risks", "risk"),
        (r"what deadlines are dangerous", "risk"),
        (r"what is coming up", "risk"),
        (r"(?:show )?deadline risk", "risk"),
        (r"(?:show )?task health(?: scores?)?", "health"),
        (r"(?:show )?(?:the )?deadline heatmap", "heatmap"),
        (r"where are (?:the )?bottlenecks", "bottlenecks"),
        (r"(?:show )?(?:operational )?bottlenecks", "bottlenecks"),
        (r"which tasks are unassigned", "unassigned"),
        (r"which dates have deadline collisions", "collisions"),
        (r"when can i meet with .+", "meeting"),
        (r"find a time for the team", "meeting"),
        (r"when is everyone available", "meeting"),
    )
    for pattern, mode in patterns:
        if re.fullmatch(pattern, value):
            result = {"intent": "operations_intelligence", "operations_mode": mode}
            person = re.fullmatch(r"when can i meet with (.+)", value)
            if person:
                result["calendar_member"] = person.group(1).strip()
            return result
    return None


def workload_labels(rows):
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
            labels[owner_id] = "High"
        elif score <= max(1, median - 3):
            labels[owner_id] = "Low"
        else:
            labels[owner_id] = "Medium"
    return labels


def assess_risks(tasks, today):
    """Return transparent deterministic task risk assessments."""
    values = list(tasks)
    workloads = {row.owner_id: row for row in
                 predictive_intelligence.calculate_workload_pressure(values, today)}
    clusters = predictive_intelligence.calculate_deadline_concentration(
        values, today, minimum_tasks=2, horizon_days=30)
    cluster_by_task = {task_id: cluster for cluster in clusters for task_id in cluster.task_ids}
    risks = []
    for task in values:
        if task.completed:
            continue
        points, reasons = 0, []
        if task.due_date and task.due_date < today:
            days = (today - task.due_date).days
            points += 5
            reasons.append(f"Overdue by {days} day{'s' if days != 1 else ''}")
        elif task.due_date:
            days = (task.due_date - today).days
            if days <= 1:
                points += 3; reasons.append("Due within 24 hours")
            elif days <= 7:
                points += 1; reasons.append(f"Due in {days} days")
        if task.priority == "P1":
            points += 3; reasons.append("P1 priority")
        elif task.priority == "P2":
            points += 1; reasons.append("P2 priority")
        if not task.owner_ids:
            points += 2; reasons.append("No assigned owner")
        owner = workloads.get(task.owner_ids[0]) if len(task.owner_ids) == 1 else None
        if owner and owner.p1 >= 3:
            points += 1; reasons.append(f"Owner has {owner.p1} pending P1 tasks")
        cluster = cluster_by_task.get(task.item_id)
        if cluster:
            points += 2; reasons.append(f"{len(cluster.task_ids)} tasks share this deadline")
        level = "Critical" if points >= 8 else "High" if points >= 5 else "Medium" if points >= 2 else "Low"
        risks.append(TaskRisk(task, level, points, tuple(reasons or ("No immediate pressure signal",))))
    rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    return tuple(sorted(risks, key=lambda row: (
        rank[row.level], row.task.due_date or date.max, row.task.name.casefold())))


def task_health(risk):
    return "Critical" if risk.level == "Critical" else "At Risk" if risk.level in {"High", "Medium"} else "Healthy"


def heatmap(tasks, today, horizon_days=31):
    grouped = defaultdict(list)
    end = today + timedelta(days=horizon_days)
    for task in tasks:
        if not task.completed and task.due_date and today <= task.due_date <= end:
            grouped[task.due_date].append(task)
    return tuple((day, len(values), sum(task.priority == "P1" for task in values))
                 for day, values in sorted(grouped.items()))


def bottlenecks(tasks, today, name_for_user):
    values = list(tasks)
    summary = predictive_intelligence.build_predictive_summary(values, today)
    labels = workload_labels(summary.workload)
    results = []
    for row in summary.workload:
        if row.owner_id and labels.get(row.owner_id) == "High":
            results.append(Bottleneck(
                name_for_user(row.owner_id),
                f"{row.pending} pending, {row.overdue} overdue, {row.p1} P1 tasks.", "High"))
    unassigned_p1 = sum(not task.owner_ids and task.priority == "P1" and not task.completed
                        for task in values)
    if unassigned_p1:
        results.append(Bottleneck(
            "Unassigned high-priority work",
            f"{unassigned_p1} pending P1 task{'s have' if unassigned_p1 != 1 else ' has'} no owner.",
            "High"))
    for cluster in predictive_intelligence.calculate_deadline_concentration(
            values, today, minimum_tasks=3, horizon_days=30):
        p1 = cluster.priority_counts.get("P1", 0)
        results.append(Bottleneck(
            f"Deadline cluster — {cluster.due_date.strftime('%d %b %Y')}",
            f"{len(cluster.task_ids)} tasks share this date, including {p1} P1.",
            "High" if p1 else "Medium"))
    return tuple(results)


def meeting_windows(clocks, requester_clock=None, duration_minutes=60):
    """Find overlap in configured working hours, expressed in requester time."""
    usable = [clock for clock in clocks if clock.local_time and clock.working_start and clock.working_end]
    if len(usable) != len(clocks) or not usable:
        return ()
    requester = requester_clock or usable[0]
    day = requester.local_time.date()
    starts, ends = [], []
    for clock in usable:
        local_start = datetime.combine(clock.local_time.date(), clock.working_start,
                                       tzinfo=clock.local_time.tzinfo)
        local_end = datetime.combine(clock.local_time.date(), clock.working_end,
                                     tzinfo=clock.local_time.tzinfo)
        starts.append(local_start.astimezone(requester.local_time.tzinfo))
        ends.append(local_end.astimezone(requester.local_time.tzinfo))
    start, end = max(starts), min(ends)
    if start.date() != day or end <= start or end - start < timedelta(minutes=duration_minutes):
        return ()
    return ((start, min(end, start + timedelta(minutes=duration_minutes))),)
