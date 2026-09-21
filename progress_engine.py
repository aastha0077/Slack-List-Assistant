"""Deterministic progress calculations over Slack List item snapshots.

This module never calls Slack and never interprets language. Callers provide
already-authorized List records, schema metadata, requested metrics and dates.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable

import slack_tools
import visualization


METRICS = {
    "overview", "completion", "workload", "status_distribution",
    "priority_distribution", "overdue", "due_today", "due_this_week",
    "upcoming", "at_risk", "completed_over_time", "created_over_time",
    "comparison", "summary",
}


@dataclass
class ProgressReport:
    requested: tuple[str, ...]
    snapshot: dict = field(default_factory=dict)
    workload: dict = field(default_factory=dict)
    status_distribution: dict = field(default_factory=dict)
    priority_distribution: dict = field(default_factory=dict)
    overdue_workload: dict = field(default_factory=dict)
    overdue_items: list = field(default_factory=list)
    due_today_items: list = field(default_factory=list)
    due_this_week_items: list = field(default_factory=list)
    upcoming_items: list = field(default_factory=list)
    at_risk_items: list = field(default_factory=list)
    completed_series: dict = field(default_factory=dict)
    created_series: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    displayed_items: list = field(default_factory=list)


def _parse_date(value):
    """Return a date only when Slack supplied a recognizable date/timestamp."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)):
        for part in value:
            parsed = _parse_date(part)
            if parsed:
                return parsed
        return None
    if isinstance(value, dict):
        for key in ("timestamp", "date", "datetime", "value", "text"):
            parsed = _parse_date(value.get(key))
            if parsed:
                return parsed
        return None
    if isinstance(value, (int, float)) or str(value).strip().replace(".", "", 1).isdigit():
        try:
            raw = float(value)
            if raw > 10_000_000_000:
                raw /= 1000
            return datetime.fromtimestamp(raw, timezone.utc).date()
        except (ValueError, TypeError, OSError, OverflowError):
            return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None


def _timestamp_from_schema(item, schema, dimension):
    for field in slack_tools._schema_fields(schema):
        label = " ".join(str(field.get(key) or "") for key in ("key", "name")).casefold()
        if dimension not in label or not any(word in label for word in ("date", "time", " at")):
            continue
        parsed = _parse_date(slack_tools.extract_field_value(item, schema, field.get("key") or field.get("name")))
        if parsed:
            return parsed
    return None


def completion_date(item, schema):
    for key in ("completed_at", "completion_timestamp", "completed_timestamp", "date_completed"):
        parsed = _parse_date(item.get(key))
        if parsed:
            return parsed
    return _timestamp_from_schema(item, schema, "complet")


def creation_date(item, schema):
    for key in ("date_created", "created_timestamp", "created_at", "creation_timestamp"):
        parsed = _parse_date(item.get(key))
        if parsed:
            return parsed
    return _timestamp_from_schema(item, schema, "creat")


def _is_completed(item, schema):
    completed_column = slack_tools.column(
        schema, keys=slack_tools.COMPLETED_KEYS,
        names={"Completed"}, types={"todo_completed", "completed", "checkbox"})
    if completed_column:
        return slack_tools.extract_completed(item, schema)
    return slack_tools.extract_status(item, schema) == "completed"


def _due(item, schema):
    return _parse_date(slack_tools.extract_due_date(item, schema))


def calculate_completion_rate(tasks, schema):
    total = len(tasks)
    completed = sum(1 for item in tasks if _is_completed(item, schema))
    return round(completed * 100 / total, 1) if total else 0.0


def calculate_workload(tasks, schema, name_for_user: Callable[[str], str], today=None):
    workload = defaultdict(lambda: {"total": 0, "pending": 0, "completed": 0, "overdue": 0})
    today = today or date.today()
    for item in tasks:
        owners = slack_tools.extract_assignee_ids(item, schema) or [None]
        done = _is_completed(item, schema)
        overdue = bool(not done and _due(item, schema) and _due(item, schema) < today)
        for owner in owners:
            label = name_for_user(owner) if owner else "Unassigned"
            row = workload[label]
            row["total"] += 1
            row["completed" if done else "pending"] += 1
            row["overdue"] += int(overdue)
    return dict(sorted(workload.items(), key=lambda pair: (-pair[1]["pending"], pair[0].casefold())))


def calculate_priority_distribution(tasks, schema):
    values = Counter(slack_tools.extract_priority(item, schema) or "Unspecified" for item in tasks)
    order = ("P1", "P2", "P3", "P4", "Unspecified")
    return {key: values[key] for key in order if values[key]}


def calculate_status_distribution(tasks, schema, today=None, include_overdue=True,
                                  requested_statuses=None):
    """Return mutually exclusive completion, pending and overdue buckets."""
    today = today or date.today()
    values = Counter()
    for item in tasks:
        if _is_completed(item, schema):
            values["Completed"] += 1
        elif include_overdue and _due(item, schema) and _due(item, schema) < today:
            values["Overdue"] += 1
        else:
            values["Pending"] += 1
    if requested_statuses:
        binary = Counter()
        for item in tasks:
            binary["Completed" if _is_completed(item, schema) else "Pending"] += 1
        labels = ["Completed" if value == "completed" else "Pending"
                  for value in requested_statuses]
        return {label: binary[label] for label in dict.fromkeys(labels)}
    return {key: values[key] for key in ("Completed", "Pending", "Overdue") if values[key]}


def calculate_time_series(tasks, schema, dimension, period=None):
    extractor = completion_date if dimension == "completed" else creation_date
    counts = Counter()
    available = 0
    missing = 0
    start = _parse_date((period or {}).get("start"))
    end = _parse_date((period or {}).get("end"))
    for item in tasks:
        if dimension == "completed" and not _is_completed(item, schema):
            continue
        observed = extractor(item, schema)
        if not observed:
            missing += 1
            continue
        available += 1
        if (start and observed < start) or (end and observed > end):
            continue
        counts[observed.isoformat()] += 1
    return {"values": dict(sorted(counts.items())), "available": available, "missing": missing}


def calculate_progress(tasks, schema, today=None, name_for_user=None, available_fields=None,
                       metrics=None, period=None, comparison_periods=None,
                       requested_statuses=None):
    """Calculate requested metrics from an authorized, current List snapshot."""
    tasks = list(tasks)
    today = today or date.today()
    name_for_user = name_for_user or (lambda user_id: user_id)
    if available_fields is None:
        available_fields = {"status", "due_date", "priority", "assignee"}
    requested = tuple(dict.fromkeys(metrics or ("overview",)))
    requested_set = set(requested)
    wants_summary = "summary" in requested_set
    report = ProgressReport(requested=requested)
    completed = [item for item in tasks if _is_completed(item, schema)] if "status" in available_fields else []
    pending = [item for item in tasks if not _is_completed(item, schema)] if "status" in available_fields else []
    dated = [(item, _due(item, schema)) for item in tasks] if "due_date" in available_fields else []
    overdue = [item for item, due in dated if due and due < today and item in pending]
    due_today = [item for item, due in dated if due == today and item in pending]
    week_end = today + timedelta(days=6 - today.weekday())
    due_week = [item for item, due in dated if due and today <= due <= week_end and item in pending]

    report.snapshot = {
        "total": len(tasks), "completed": len(completed), "pending": len(pending),
        "overdue": len(overdue), "due_today": len(due_today), "due_this_week": len(due_week),
        "completion_rate": calculate_completion_rate(tasks, schema) if "status" in available_fields else None,
    }
    if "status" not in available_fields and (
            wants_summary or requested_set & {
                "overview", "completion", "workload", "status_distribution",
                "at_risk", "completed_over_time"}):
        report.limitations.append("Task completion/status is unavailable or not readable for this List.")
    if "due_date" not in available_fields and (
            wants_summary or requested_set & {
                "overview", "overdue", "due_today", "due_this_week", "upcoming", "at_risk"}):
        report.limitations.append("Due-date metrics are unavailable or not readable for this List.")

    wants_workload = wants_summary or "workload" in requested_set
    if wants_workload and {"assignee", "status"}.issubset(available_fields):
        report.workload = calculate_workload(tasks, schema, name_for_user, today)
        report.overdue_workload = {name: values["overdue"] for name, values in report.workload.items()
                                   if values["overdue"]}
    elif wants_workload and "assignee" not in available_fields:
        report.limitations.append("Assignee workload is unavailable or not readable for this List.")
    elif wants_workload:
        report.limitations.append("Pending workload requires a readable completion/status field.")
    if "priority_distribution" in requested_set and "priority" in available_fields:
        report.priority_distribution = calculate_priority_distribution(tasks, schema)
    elif "priority_distribution" in requested_set:
        report.limitations.append("Priority distribution is unavailable or not readable for this List.")
    if "status_distribution" in requested_set and "status" in available_fields:
        report.status_distribution = calculate_status_distribution(
            tasks, schema, today, include_overdue="due_date" in available_fields,
            requested_statuses=requested_statuses)

    report.overdue_items = overdue
    report.due_today_items = due_today
    report.due_this_week_items = due_week
    report.upcoming_items = [item for item, due in sorted(dated, key=lambda pair: pair[1] or date.max)
                             if due and due >= today and item in pending]
    if wants_summary or "at_risk" in requested_set:
        for item in pending:
            due = _due(item, schema) if "due_date" in available_fields else None
            priority = slack_tools.extract_priority(item, schema) if "priority" in available_fields else None
            if (due and due <= today + timedelta(days=3)) or priority == "P1":
                report.at_risk_items.append(item)
    if (wants_summary or "at_risk" in requested_set) and (
            "status" not in available_fields or not ({"due_date", "priority"} & set(available_fields))):
        report.limitations.append(
            "At-risk tasks require readable status and due-date or priority fields.")

    if "completed_over_time" in requested_set or wants_summary:
        report.completed_series = (calculate_time_series(tasks, schema, "completed", period)
                                   if "status" in available_fields
                                   else {"values": {}, "available": 0, "missing": 0})
        if not report.completed_series["available"]:
            report.limitations.append(
                "Slack List does not expose reliable completion timestamps for these tasks, so completed work over time cannot be calculated.")
    if "created_over_time" in requested_set:
        report.created_series = calculate_time_series(tasks, schema, "created", period)
        if not report.created_series["available"]:
            report.limitations.append(
                "Slack List does not expose reliable creation timestamps for these tasks, so created work over time cannot be calculated.")
    if "comparison" in requested_set:
        current_series = calculate_time_series(
            tasks, schema, "completed", (comparison_periods or {}).get("current"))
        previous_series = calculate_time_series(
            tasks, schema, "completed", (comparison_periods or {}).get("previous"))
        report.comparison = {
            "current": sum(current_series["values"].values()),
            "previous": sum(previous_series["values"].values()),
            "available": max(current_series["available"], previous_series["available"]),
        }
        if not report.comparison["available"]:
            report.limitations.append(
                "Slack List does not expose reliable completion timestamps, so period comparison cannot be calculated.")
    return report


def render_progress(report: ProgressReport, format_items: Callable[[Iterable, str], str]):
    """Backward-compatible entry point for the modular Slack renderer."""
    return visualization.render_progress(report, format_items)
