"""Deterministic prepared actions layered on persisted Sentinel alerts."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class AutopilotRecommendation:
    recommendation_id: str
    alert_id: str
    task_id: str
    requesting_user: str
    target_user: str | None
    action_type: str
    task_state_version: str
    created_at: float
    status: str
    recommendation: str
    prepared_message: str | None
    executable: bool


def _deadline_phrase(due_date: date | None, today: date) -> str:
    if due_date is None:
        return "has no due date"
    delta = (due_date - today).days
    if delta == -1:
        return "was due yesterday"
    if delta < -1:
        return f"was due {-delta} days ago"
    if delta == 0:
        return "is due today"
    if delta == 1:
        return "is due tomorrow"
    return f"is due on {due_date.strftime('%b %-d')}"


def prepare_recommendation(*, alert_id: str, payload: dict, requesting_user: str,
                           owner_name: str | None, today: date, created_at: float,
                           status: str = "prepared") -> AutopilotRecommendation:
    """Prepare one factual next action from a Sentinel alert payload."""
    task_ids = tuple(payload.get("task_ids") or ())
    task_id = task_ids[0] if len(task_ids) == 1 else str(payload.get("task_id") or "")
    owners = tuple(payload.get("owner_ids") or ())
    # The existing executor may notify every explicitly assigned owner. The
    # first owner is the stable primary target recorded on the recommendation.
    target_user = owners[0] if owners else None
    risk_type = str(payload.get("risk_type") or "")
    task_name = str(payload.get("task_name") or "Action item")
    raw_due = payload.get("due_date")
    try:
        due_date = date.fromisoformat(str(raw_due)[:10]) if raw_due else None
    except ValueError:
        due_date = None

    if risk_type == "combined_workload_risk":
        count = len(task_ids)
        action_type = "review_workload"
        recommendation = (
            f"{owner_name or 'This owner'} — review the {count} P1 "
            f"task{'s' if count != 1 else ''} and confirm "
            f"{'their deadlines' if count != 1 else 'its deadline'}.")
        prepared_message = None
        executable = False
    elif risk_type == "unassigned_deadline_risk" or not owners:
        action_type = "assign_owner"
        recommendation = "Assign an owner or review the deadline."
        prepared_message = None
        executable = False
    elif risk_type == "overdue":
        action_type = "send_reminder"
        recommendation = f"Send a reminder to {owner_name or 'the owner'}."
        prepared_message = (
            f"Hi {owner_name or 'there'}, {task_name} is still pending and "
            f"{_deadline_phrase(due_date, today)}. Please confirm the status or update the deadline.")
        executable = True
    elif risk_type == "deadline_risk":
        action_type = "send_deadline_reminder"
        recommendation = f"Send a deadline reminder to {owner_name or 'the owner'}."
        prepared_message = (
            f"Hi {owner_name or 'there'}, {task_name} {_deadline_phrase(due_date, today)} "
            "and is still pending. Please confirm the completion status or update the deadline.")
        executable = True
    else:
        action_type = "review_task"
        recommendation = "No safe automated recommendation is available for this risk."
        prepared_message = None
        executable = False

    state_version = str(payload.get("task_state_hash") or "")
    recommendation_id = hashlib.sha256(
        f"{alert_id}|{requesting_user}|{action_type}|{state_version}".encode()).hexdigest()
    return AutopilotRecommendation(
        recommendation_id=recommendation_id, alert_id=alert_id, task_id=task_id,
        requesting_user=requesting_user, target_user=target_user,
        action_type=action_type, task_state_version=state_version,
        created_at=created_at, status=status, recommendation=recommendation,
        prepared_message=prepared_message, executable=executable)
