"""Deterministic project-management insights over authorized Slack List data.

The functions in this module are deliberately unaware of Slack transport,
conversation state, RBAC and language parsing.  Callers supply current records
and schema metadata; this module only calculates explainable results.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import mean
from typing import Callable, Iterable

import slack_tools
from progress_engine import completion_date


PRIORITY_RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}


@dataclass(frozen=True)
class TaskHealth:
    item_id: str
    item: dict
    level: str
    icon: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PlanEntry:
    item_id: str
    item: dict
    scheduled_date: date
    reasons: tuple[str, ...]


@dataclass
class WorkloadReport:
    rows: dict = field(default_factory=dict)
    overloaded: list[str] = field(default_factory=list)
    suggestions: list[dict] = field(default_factory=list)


@dataclass
class StandupReport:
    completed: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    attention: list[TaskHealth] = field(default_factory=list)
    upcoming: list = field(default_factory=list)
    completion_is_daily: bool = False
    limitations: list[str] = field(default_factory=list)


def _as_date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _schema_fields(schema):
    if isinstance(schema, list):
        return schema
    if not isinstance(schema, dict):
        return []
    return schema.get("schema") or schema.get("fields") or schema.get("columns") or []


def dependency_fields(schema):
    """Discover explicit dependency/blocking fields without inventing schema."""
    discovered = []
    for field_value in _schema_fields(schema):
        label = " ".join(str(field_value.get(key) or "") for key in ("key", "name", "title")).casefold()
        if any(token in label for token in ("depend", "blocked by", "blocker", "blocking")):
            discovered.append(field_value)
    return discovered


def dependency_values(item, schema):
    values = []
    for schema_field in dependency_fields(schema):
        key = schema_field.get("key") or schema_field.get("name") or schema_field.get("id")
        value = slack_tools.extract_field_value(item, schema, key)
        if value not in (None, "", [], False):
            values.append(str(value))
    return tuple(values)


def classify_task(item, schema, today=None, attention_days=3):
    """Classify one task from factual fields and retain every reason."""
    today = today or date.today()
    item_id = slack_tools.extract_item_id(item)
    completed = slack_tools.extract_completed(item, schema)
    due = _as_date(slack_tools.extract_due_date(item, schema))
    priority = slack_tools.extract_priority(item, schema)
    dependencies = dependency_values(item, schema)
    reasons = []

    if completed:
        return TaskHealth(item_id, item, "On Track", "🟢", ("Completed",))
    if due and due < today:
        days = (today - due).days
        reasons.append(f"Overdue by {days} day{'s' if days != 1 else ''}")
        if priority:
            reasons.append(priority)
        if dependencies:
            reasons.append("Explicit dependency/blocker data is present")
        return TaskHealth(item_id, item, "Overdue", "🔴", tuple(reasons))
    if due is None:
        reasons.append("No due date")
        if priority:
            reasons.append(priority)
        if dependencies:
            reasons.append("Explicit dependency/blocker data is present")
        return TaskHealth(item_id, item, "No Deadline", "⚪", tuple(reasons))

    days = (due - today).days
    if days == 0:
        reasons.append("Due today")
    elif days == 1:
        reasons.append("Due tomorrow")
    elif days <= attention_days:
        reasons.append(f"Due in {days} days")
    if priority == "P1":
        reasons.append("P1 priority")
    if dependencies:
        reasons.append("Explicit dependency/blocker data is present")
    if reasons:
        reasons.append("Still pending")
        return TaskHealth(item_id, item, "Needs Attention", "🟡", tuple(reasons))
    return TaskHealth(item_id, item, "On Track", "🟢", (f"Due {due.isoformat()}", priority or "No priority"))


def calculate_health(tasks, schema, today=None, attention_only=False, attention_days=3):
    records = [classify_task(item, schema, today, attention_days) for item in tasks]
    if attention_only:
        records = [record for record in records if record.level in {"Needs Attention", "Overdue"}]
    order = {"Overdue": 0, "Needs Attention": 1, "No Deadline": 2, "On Track": 3}
    return sorted(records, key=lambda record: (order[record.level], slack_tools.extract_item_name(record.item, schema).casefold()))


def _weekdays(start, end):
    days = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def build_plan(tasks, schema, start, end, today=None):
    """Create a due-date proposal; never mutates records."""
    today = today or date.today()
    days = _weekdays(start, end)
    if not days:
        raise ValueError("The planning period does not contain a working day.")
    pending = [item for item in tasks if not slack_tools.extract_completed(item, schema)]

    def rank(item):
        due = _as_date(slack_tools.extract_due_date(item, schema))
        return (
            0 if due and due < today else 1,
            due or date.max,
            PRIORITY_RANK.get(slack_tools.extract_priority(item, schema), 5),
            slack_tools.extract_item_name(item, schema).casefold(),
        )

    entries = []
    for index, item in enumerate(sorted(pending, key=rank)):
        original_due = _as_date(slack_tools.extract_due_date(item, schema))
        scheduled = days[min(index, len(days) - 1)]
        reasons = []
        if original_due and original_due < today:
            reasons.append("currently overdue")
        elif original_due:
            reasons.append(f"current deadline {original_due.isoformat()}")
        else:
            reasons.append("no current deadline")
        priority = slack_tools.extract_priority(item, schema)
        if priority:
            reasons.append(priority)
        entries.append(PlanEntry(
            slack_tools.extract_item_id(item), item, scheduled, tuple(reasons)))
    return entries


def calculate_workload(tasks, schema, name_for_user: Callable[[str], str], today=None,
                       eligible_user_ids: Iterable[str] = ()): 
    """Calculate workload and deterministic rebalance suggestions."""
    today = today or date.today()
    by_user = {user_id: [] for user_id in eligible_user_ids}
    unassigned = []
    for item in tasks:
        if slack_tools.extract_completed(item, schema):
            continue
        owners = slack_tools.extract_assignee_ids(item, schema)
        if not owners:
            unassigned.append(item)
        for owner in owners:
            by_user.setdefault(owner, []).append(item)

    rows = {}
    scores = {}
    for user_id, owned in by_user.items():
        overdue = sum(1 for item in owned if (_as_date(slack_tools.extract_due_date(item, schema)) or date.max) < today)
        p1 = sum(1 for item in owned if slack_tools.extract_priority(item, schema) == "P1")
        upcoming = sum(1 for item in owned if (
            (due := _as_date(slack_tools.extract_due_date(item, schema))) is not None
            and today <= due <= today + timedelta(days=7)))
        score = len(owned) + 2 * overdue + 2 * p1
        rows[user_id] = {
            "name": name_for_user(user_id), "pending": len(owned), "overdue": overdue,
            "p1": p1, "upcoming": upcoming, "score": score,
        }
        scores[user_id] = score
    if unassigned:
        rows[None] = {"name": "Unassigned", "pending": len(unassigned), "overdue": 0,
                      "p1": 0, "upcoming": 0, "score": len(unassigned)}

    active_scores = list(scores.values())
    average = mean(active_scores) if active_scores else 0
    overloaded = [user_id for user_id, score in scores.items()
                  if score > average * 1.25 and score >= average + 2]
    suggestions = []
    if scores:
        recipients = sorted(scores, key=lambda user_id: (scores[user_id], name_for_user(user_id).casefold()))
        for source in sorted(overloaded, key=lambda user_id: -scores[user_id]):
            destination = next((user_id for user_id in recipients if user_id != source), None)
            if not destination or scores[source] <= scores[destination] + 2:
                continue
            movable = sorted(
                by_user[source],
                key=lambda item: (
                    PRIORITY_RANK.get(slack_tools.extract_priority(item, schema), 5),
                    _as_date(slack_tools.extract_due_date(item, schema)) or date.max,
                ), reverse=True)
            if movable:
                item = movable[0]
                suggestions.append({
                    "item_id": slack_tools.extract_item_id(item), "item": item,
                    "from_user_id": source, "to_user_id": destination,
                    "reason": f"workload score {scores[source]} versus {scores[destination]}",
                })
    return WorkloadReport(rows=rows, overloaded=overloaded, suggestions=suggestions)


def build_standup(tasks, schema, today=None):
    today = today or date.today()
    completed = [item for item in tasks if slack_tools.extract_completed(item, schema)]
    timestamped = [(item, completion_date(item, schema)) for item in completed]
    available = [pair for pair in timestamped if pair[1] is not None]
    report = StandupReport()
    if available:
        report.completed = [item for item, completed_on in available if completed_on == today]
        report.completion_is_daily = True
        if len(available) != len(completed):
            report.limitations.append(
                "Some completed tasks have no completion timestamp and are excluded from today's completed section.")
    else:
        report.completed = completed
        report.limitations.append(
            "Slack List does not expose reliable completion timestamps, so completed tasks are shown as current state, not as completed today.")
    report.pending = [item for item in tasks if not slack_tools.extract_completed(item, schema)]
    report.attention = calculate_health(report.pending, schema, today, attention_only=True)
    tomorrow = today + timedelta(days=1)
    report.upcoming = [item for item in report.pending
                       if _as_date(slack_tools.extract_due_date(item, schema)) == tomorrow]
    return report
