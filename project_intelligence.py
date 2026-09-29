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
from progress_engine import completion_date, creation_date


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


@dataclass(frozen=True)
class FocusEntry:
    item_id: str
    item: dict
    due_date: date | None
    priority: str | None
    reason: str
    category: str


@dataclass(frozen=True)
class RiskEntry:
    item_id: str
    item: dict
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class HealthSummary:
    pending: int
    completed: int
    overdue: int
    p1: int
    unassigned: int
    due_within_48h: int
    risks: tuple[RiskEntry, ...]


@dataclass(frozen=True)
class NormalizedTask:
    """One immutable intelligence view over a current Slack List item."""
    item_id: str
    item: dict
    name: str
    owner_ids: tuple[str, ...]
    priority: str | None
    due_date: date | None
    completed: bool
    status: str
    created_date: date | None


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


def normalize_task(item, schema):
    """Normalize one raw Slack List item through the existing field extractors."""
    if isinstance(item, NormalizedTask):
        return item
    status = str(slack_tools.extract_status(item, schema) or "").strip()
    priority = slack_tools.normalize_priority(slack_tools.extract_priority(item, schema), schema)
    return NormalizedTask(
        item_id=slack_tools.extract_item_id(item), item=item,
        name=slack_tools.extract_item_name(item, schema) or "Unnamed task",
        owner_ids=tuple(slack_tools.extract_assignee_ids(item, schema)),
        priority=priority,
        due_date=_as_date(slack_tools.extract_due_date(item, schema)),
        completed=(slack_tools.extract_completed(item, schema)
                   or status.casefold() in {"cancelled", "canceled", "closed"}),
        status=status, created_date=creation_date(item, schema),
    )


def normalize_task_snapshot(tasks, schema):
    """Create the single normalized snapshot consumed by intelligence features."""
    values = list(tasks)
    if values and all(isinstance(item, NormalizedTask) for item in values):
        return values
    return [normalize_task(item, schema) for item in values]


def calculate_daily_focus(tasks, schema, today=None):
    """Rank pending work using factual deadline and priority dimensions."""
    today = today or date.today()
    entries = []
    for task in normalize_task_snapshot(tasks, schema):
        if task.completed:
            continue
        due = task.due_date
        priority = task.priority
        if due and due < today:
            days = (today - due).days
            reason = f"{priority + ' + ' if priority == 'P1' else ''}overdue by {days} day{'s' if days != 1 else ''}"
            deadline_rank = 0
            category = "immediate"
        elif due == today:
            reason = "high priority + due today" if priority == "P1" else "due today"
            deadline_rank = 1
            category = "immediate"
        elif due == today + timedelta(days=1):
            reason = "high priority + approaching deadline" if priority == "P1" else "due tomorrow"
            deadline_rank = 2
            category = "upcoming"
        elif due:
            # The due date is rendered as its own human-readable field. Keep
            # the explanation concise instead of repeating an internal ISO date.
            reason = "upcoming deadline"
            deadline_rank = 3
            category = "upcoming"
        else:
            reason = "pending with no due date"
            deadline_rank = 4
            category = "upcoming"
        entries.append((
            deadline_rank,
            PRIORITY_RANK.get(priority, 5),
            due or date.max,
            task.name.casefold(),
            FocusEntry(task.item_id, task.item, due, priority, reason, category),
        ))
    return [entry[-1] for entry in sorted(entries, key=lambda value: value[:-1])]


def analyze_task_risks(tasks, schema, today=None):
    """Return explainable risk signals without assigning a subjective score."""
    today = today or date.today()
    assignee_available = bool(slack_tools.column(
        schema, keys=slack_tools.ASSIGNEE_KEYS, names={"Assignee", "Owner"}))
    risks = []
    normalized = normalize_task_snapshot(tasks, schema)
    for task in normalized:
        if task.completed:
            continue
        due = task.due_date
        priority = task.priority
        owners = task.owner_ids
        reasons = []
        if due and due < today:
            days = (today - due).days
            reasons.append(f"Overdue by {days} day{'s' if days != 1 else ''}")
        elif due == today:
            reasons.append("Due today")
        elif due and due <= today + timedelta(days=2):
            hours = (due - today).days * 24
            reasons.append(f"Due within {hours} hours")
        if priority == "P1":
            reasons.append("P1 priority")
        if assignee_available and not owners:
            reasons.append("No assigned owner")
        if reasons:
            risks.append(RiskEntry(task.item_id, task.item, tuple(reasons)))

    by_id = {task.item_id: task for task in normalized}
    def rank(record):
        task = by_id[record.item_id]
        due = task.due_date
        return (
            0 if due and due < today else 1,
            0 if task.priority == "P1" else 1,
            due or date.max,
            task.name.casefold(),
        )
    return sorted(risks, key=rank)


def generate_task_health(tasks, schema, today=None):
    """Calculate aggregate health facts from one current task snapshot."""
    today = today or date.today()
    normalized = normalize_task_snapshot(tasks, schema)
    completed = [task for task in normalized if task.completed]
    pending = [task for task in normalized if not task.completed]
    overdue = 0
    due_within_48h = 0
    for task in pending:
        due = task.due_date
        if due and due < today:
            overdue += 1
        elif due and today <= due <= today + timedelta(days=2):
            due_within_48h += 1
    risks = analyze_task_risks(pending, schema, today)
    assignee_available = bool(slack_tools.column(
        schema, keys=slack_tools.ASSIGNEE_KEYS, names={"Assignee", "Owner"}))
    return HealthSummary(
        pending=len(pending), completed=len(completed), overdue=overdue,
        p1=sum(1 for task in pending if task.priority == "P1"),
        unassigned=(sum(1 for task in pending if not task.owner_ids)
                    if assignee_available else 0),
        due_within_48h=due_within_48h, risks=tuple(risks),
    )


def generate_weekly_insights(tasks, schema, name_for_user, today=None):
    """Build concise, factual insight sentences from pending tasks."""
    today = today or date.today()
    normalized = normalize_task_snapshot(tasks, schema)
    health = generate_task_health(normalized, schema, today)
    insights = []
    if health.overdue:
        insights.append(f"{health.overdue} task{'s are' if health.overdue != 1 else ' is'} overdue.")
    if health.due_within_48h:
        insights.append(
            f"{health.due_within_48h} task{'s are' if health.due_within_48h != 1 else ' is'} due within 48 hours.")
    if health.unassigned:
        insights.append(
            f"{health.unassigned} pending task{'s are' if health.unassigned != 1 else ' is'} unassigned.")
    p1_by_owner = {}
    for task in normalized:
        if task.completed or task.priority != "P1":
            continue
        for owner_id in task.owner_ids:
            p1_by_owner[owner_id] = p1_by_owner.get(owner_id, 0) + 1
    for owner_id, count in sorted(p1_by_owner.items(), key=lambda pair: (-pair[1], name_for_user(pair[0]).casefold())):
        if count >= 2:
            insights.append(f"{name_for_user(owner_id)} has {count} pending P1 tasks.")
    return insights


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
    task = normalize_task(item, schema)
    item_id = task.item_id
    completed = task.completed
    due = task.due_date
    priority = task.priority
    dependencies = dependency_values(task.item, schema)
    reasons = []

    if completed:
        return TaskHealth(item_id, task.item, "On Track", "🟢", ("Completed",))
    if due and due < today:
        days = (today - due).days
        reasons.append(f"Overdue by {days} day{'s' if days != 1 else ''}")
        if priority:
            reasons.append(priority)
        if dependencies:
            reasons.append("Explicit dependency/blocker data is present")
        return TaskHealth(item_id, task.item, "Overdue", "🔴", tuple(reasons))
    if due is None:
        reasons.append("No due date")
        if priority:
            reasons.append(priority)
        if dependencies:
            reasons.append("Explicit dependency/blocker data is present")
        return TaskHealth(item_id, task.item, "No Deadline", "⚪", tuple(reasons))

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
        return TaskHealth(item_id, task.item, "Needs Attention", "🟡", tuple(reasons))
    return TaskHealth(item_id, task.item, "On Track", "🟢", (f"Due {due.isoformat()}", priority or "No priority"))


def calculate_health(tasks, schema, today=None, attention_only=False, attention_days=3):
    snapshot = normalize_task_snapshot(tasks, schema)
    records = [classify_task(task, schema, today, attention_days) for task in snapshot]
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
    normalized = normalize_task_snapshot(tasks, schema)
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
        overdue = sum(1 for task in owned if (task.due_date or date.max) < today)
        p1 = sum(1 for task in owned if task.priority == "P1")
        p2 = sum(1 for task in owned if task.priority == "P2")
        p3 = sum(1 for task in owned if task.priority == "P3")
        due_soon = sum(1 for task in owned if (
            task.due_date is not None and today <= task.due_date <= today + timedelta(days=2)))
        upcoming = sum(1 for task in owned if (
            task.due_date is not None and today <= task.due_date <= today + timedelta(days=7)))
        score = len(owned) + 2 * overdue + 2 * p1
        rows[user_id] = {
            "name": name_for_user(user_id), "pending": len(owned), "overdue": overdue,
            "p1": p1, "p2": p2, "p3": p3, "due_soon": due_soon,
            "upcoming": upcoming, "score": score,
        }
        scores[user_id] = score
    if unassigned:
        rows[None] = {
            "name": "Unassigned", "pending": len(unassigned),
            "overdue": sum(1 for task in unassigned if (task.due_date or date.max) < today),
            "p1": sum(1 for task in unassigned if task.priority == "P1"),
            "p2": sum(1 for task in unassigned if task.priority == "P2"),
            "p3": sum(1 for task in unassigned if task.priority == "P3"),
            "due_soon": sum(1 for task in unassigned if (
                task.due_date is not None and today <= task.due_date <= today + timedelta(days=2))),
            "upcoming": sum(1 for task in unassigned if (
                task.due_date is not None and today <= task.due_date <= today + timedelta(days=7))),
            "score": len(unassigned),
        }

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
                key=lambda task: (
                    PRIORITY_RANK.get(task.priority, 5),
                    task.due_date or date.max,
                ), reverse=True)
            if movable:
                task = movable[0]
                suggestions.append({
                    "item_id": task.item_id, "item": task.item,
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
