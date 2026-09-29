"""Read-only orchestration for a normalized Slack List task snapshot."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import logging

import action_item_sentinel
import project_intelligence
import predictive_intelligence


logger = logging.getLogger("slack_list.command_center")


@dataclass(frozen=True)
class CommandCenterReport:
    pending: int
    completed: int
    priorities: dict[str, int]
    overdue: int
    due_this_week: int
    critical: tuple[project_intelligence.NormalizedTask, ...]
    risks: tuple[action_item_sentinel.SentinelRisk, ...]
    workload: project_intelligence.WorkloadReport
    reminder_count: int
    workload_review_count: int
    insight: str
    next_step: str
    predictive: predictive_intelligence.PredictiveSummary


def build_report(snapshot, *, today: date, name_for_user) -> CommandCenterReport:
    """Calculate all Command Center facts from one normalized snapshot."""
    tasks = list(snapshot)
    pending = [task for task in tasks if not task.completed]
    completed = [task for task in tasks if task.completed]
    priorities = {key: sum(task.priority == key for task in pending)
                  for key in ("P1", "P2", "P3")}
    overdue = [task for task in pending if task.due_date and task.due_date < today]
    due_this_week = [task for task in pending
                     if task.due_date and today <= task.due_date <= today + timedelta(days=6)]
    critical = sorted(
        [task for task in pending if task in overdue or task.due_date == today],
        key=lambda task: (task.due_date or date.max,
                          project_intelligence.PRIORITY_RANK.get(task.priority, 5),
                          task.name.casefold()))
    risks = action_item_sentinel.detect_risks(tasks, today)
    predictive = predictive_intelligence.build_predictive_summary(tasks, today)
    try:
        workload = project_intelligence.calculate_workload(
            tasks, {}, name_for_user, today,
            {owner for task in tasks for owner in task.owner_ids})
    except Exception as exc:
        logger.error("command_center_section_degraded section=workload error_type=%s",
                     type(exc).__name__)
        workload = project_intelligence.WorkloadReport()
    reminders = sum(risk.risk_type in {"overdue", "deadline_risk"}
                    and bool(risk.owner_ids) for risk in risks)
    workload_reviews = sum(risk.risk_type == "combined_workload_risk" for risk in risks)
    unassigned_p1 = sum(task.priority == "P1" and not task.owner_ids for task in pending)
    if overdue:
        insight = "Overdue work is currently the main source of operational risk."
    elif any(risk.risk_type == "deadline_risk" for risk in risks):
        insight = "Approaching P1 deadlines are currently the main source of operational risk."
    elif workload_reviews:
        insight = "Concentrated high-priority workload is the main current risk signal."
    elif unassigned_p1:
        insight = "Unassigned high-priority work requires attention."
    else:
        insight = "No active deterministic risk signal currently dominates the task snapshot."
    if overdue:
        next_step = "Review the overdue tasks, starting with P1 items."
    elif any(risk.risk_type == "deadline_risk" for risk in risks):
        next_step = "Review the P1 tasks due within 48 hours."
    elif workload_reviews:
        next_step = "Review concentrated P1 workloads and confirm their deadlines."
    elif unassigned_p1:
        next_step = "Assign owners to the unassigned P1 tasks."
    elif pending:
        next_step = "Review the next pending deadline."
    else:
        next_step = "No pending action is required."
    return CommandCenterReport(
        len(pending), len(completed), priorities, len(overdue), len(due_this_week),
        tuple(critical[:5]), tuple(risks), workload, reminders, workload_reviews,
        insight, next_step, predictive)


def owner_risks(snapshot, owner_id: str, *, today: date):
    """Return factual owner risks without inspecting unauthorized tasks."""
    owned = [task for task in snapshot if not task.completed and owner_id in task.owner_ids]
    task_ids = {task.item_id for task in owned}
    risks = action_item_sentinel.detect_risks(snapshot, today)
    return owned, [risk for risk in risks if task_ids.intersection(risk.task_ids)
                   or owner_id in risk.owner_ids]
