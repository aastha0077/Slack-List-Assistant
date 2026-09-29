"""Controlled planning over authorized task snapshots.

This module is deliberately side-effect free.  It builds and renders plans;
the application layer owns RBAC, approval, execution, and verification through
the existing trusted services.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable

import action_item_sentinel
import project_intelligence


PLAN_TTL_SECONDS = 1800


@dataclass(frozen=True)
class OrchestratorStep:
    step_id: str
    action_type: str
    target: str
    target_task_id: str | None
    reason: str
    evidence: tuple[str, ...] = ()
    risk_level: str = "low"
    requires_approval: bool = False
    authorized: bool = False
    executable: bool = False
    status: str = "proposed"
    execution_result: str | None = None
    task_fingerprint: str | None = None
    target_user_ids: tuple[str, ...] = ()
    prepared_message: str | None = None


@dataclass(frozen=True)
class OrchestratorPlan:
    plan_id: str
    goal: str
    requester_id: str
    created_at: float
    expires_at: float
    status: str
    context: dict
    steps: tuple[OrchestratorStep, ...]
    dependencies: tuple[str, ...] = ()
    expected_outcome: str = "A reviewed, actionable task plan."
    validation_result: dict = field(default_factory=dict)
    approval_status: str = "not_requested"
    execution_status: str = "not_started"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "OrchestratorPlan":
        data = dict(value)
        normalized_steps = []
        for step in data.get("steps", ()):
            step = dict(step)
            step["evidence"] = tuple(step.get("evidence", ()))
            step["target_user_ids"] = tuple(step.get("target_user_ids", ()))
            normalized_steps.append(OrchestratorStep(**step))
        data["steps"] = tuple(normalized_steps)
        data["dependencies"] = tuple(data.get("dependencies", ()))
        return cls(**data)


def _goal_terms(goal: str) -> set[str]:
    ignored = {"prepare", "everything", "needed", "need", "help", "team", "ready",
               "current", "actions", "action", "steps", "next", "what", "should",
               "before", "for", "the", "our", "work", "get", "make", "reduce"}
    return {word for word in re.findall(r"[a-z0-9]+", goal.casefold())
            if len(word) > 2 and word not in ignored}


def resolve_goal_context(goal: str) -> str:
    """Map known task concepts to existing deterministic analysis contexts."""
    value = re.sub(r"\s+", " ", str(goal or "").casefold()).strip(" .?!")
    mappings = (
        ("overdue_work", r"\b(?:current(?:ly)?\s+)?overdue\s+(?:work|tasks?|items?)\b|\bwhat\s+is\s+(?:currently\s+)?overdue\b"),
        ("due_today", r"\b(?:tasks?|items?|work)\s+due\s+today\b|\bwhat\s+is\s+due\s+today\b"),
        ("due_24h", r"\bwithin\s+(?:the\s+next\s+)?24\s+hours?\b"),
        ("due_48h", r"\bwithin\s+(?:the\s+next\s+)?48\s+hours?\b"),
        ("deadline_risk", r"\bdue\s+soon\b|\bapproaching\s+deadline"),
        ("priority_p1", r"\b(?:p1|highest[- ]priority|critical)\s+(?:work|tasks?|items?)\b"),
        ("unassigned_tasks", r"\bunassigned\s+(?:work|tasks?|items?)\b"),
        ("current_risks", r"\bcurrent\s+risks?\b|\breduce\s+(?:the\s+)?(?:current\s+)?risks?\b"),
        ("team_workload", r"\bteam\s+workload\b|\bcurrent\s+workload\b"),
    )
    for context_type, pattern in mappings:
        if re.search(pattern, value):
            return context_type
    return "relevant_work"


def build_plan(*, goal: str, requester_id: str,
               tasks: Iterable[project_intelligence.NormalizedTask],
               risks: Iterable[action_item_sentinel.SentinelRisk],
               fingerprints: dict[str, str], context_type: str = "relevant_work",
               relevant_tasks: Iterable[project_intelligence.NormalizedTask] | None = None,
               recommendations: dict[str, object] | None = None, now: float | None = None,
               ttl_seconds: int = PLAN_TTL_SECONDS) -> OrchestratorPlan:
    """Build a bounded factual plan without authorizing or executing it."""
    now = time.time() if now is None else now
    task_values = tuple(tasks)
    pending = tuple(task for task in task_values if not task.completed)
    if relevant_tasks is not None:
        relevant = tuple(relevant_tasks)
    else:
        terms = _goal_terms(goal)
        relevant = tuple(task for task in pending if terms & set(re.findall(
            r"[a-z0-9]+", task.name.casefold()))) or pending
    relevant_ids = {task.item_id for task in relevant}
    relevant_risks = tuple(risk for risk in risks if any(
        task_id in relevant_ids for task_id in risk.task_ids))
    seed = f"{requester_id}|{goal}|{now:.6f}"
    plan_id = hashlib.sha256(seed.encode()).hexdigest()[:8]
    steps = []
    by_id = {task.item_id: task for task in relevant}
    planned_risk_task_ids = set()
    aggregate_insights = []
    aggregate_seen = set()
    for risk in relevant_risks:
        if len(risk.task_ids) != 1:
            identity = (risk.risk_type, risk.recommendation.casefold())
            if identity not in aggregate_seen:
                aggregate_seen.add(identity)
                aggregate_insights.append({
                    "identity": ":".join(identity),
                    "risk_type": risk.risk_type,
                    "message": ("Multiple urgent action items are creating workload pressure."
                                if risk.risk_type == "combined_workload_risk"
                                else risk.recommendation),
                    "risk_level": risk.severity,
                })
            continue
        if len(planned_risk_task_ids) >= 5:
            continue
        task = by_id.get(risk.task_ids[0])
        if not task:
            continue
        planned_risk_task_ids.add(task.item_id)
        owners = task.owner_ids
        recommendation = (recommendations or {}).get(task.item_id)
        action = (getattr(recommendation, "action_type", None)
                  or ("send_reminder" if owners and risk.risk_type in {"overdue", "deadline_risk"}
                      else "review_owner_assignment"))
        executable = bool(getattr(recommendation, "executable", action == "send_reminder"))
        steps.append(OrchestratorStep(
            step_id=f"{plan_id}-{len(steps) + 1}", action_type=action,
            target=task.name, target_task_id=task.item_id,
            reason=getattr(recommendation, "recommendation", None) or risk.recommendation,
            evidence=risk.reasons, risk_level=risk.severity,
            requires_approval=executable, authorized=False, executable=executable,
            task_fingerprint=fingerprints.get(task.item_id), target_user_ids=owners,
            prepared_message=getattr(recommendation, "prepared_message", None)))
    for task in relevant:
        if task.item_id in planned_risk_task_ids:
            continue
        steps.append(OrchestratorStep(
            step_id=f"{plan_id}-{len(steps) + 1}", action_type="review_task",
            target=task.name, target_task_id=task.item_id,
            reason="Review the current task status.", risk_level="low",
            authorized=True, executable=False, status="ready",
            task_fingerprint=fingerprints.get(task.item_id), target_user_ids=task.owner_ids))
    context = {
        "context_type": context_type,
        "task_count": len(task_values), "pending_count": len(pending),
        "relevant_count": len(relevant), "risk_count": len(relevant_risks),
        "relevant_task_ids": sorted(relevant_ids),
        "team_insights": aggregate_insights,
    }
    return OrchestratorPlan(
        plan_id, goal.strip(), requester_id, now, now + ttl_seconds, "proposed",
        context, tuple(steps), expected_outcome="Review the relevant risks and execute only approved, current actions.")


def validate_plan(plan: OrchestratorPlan, authorize) -> OrchestratorPlan:
    """Apply deterministic application-owned policy decisions to every step."""
    validated = []
    for step in plan.steps:
        if not step.executable:
            validated.append(replace(step, authorized=True, status="ready"))
            continue
        allowed, reason = authorize(step)
        validated.append(replace(
            step, authorized=bool(allowed), status="ready" if allowed else "unauthorized",
            execution_result=None if allowed else reason))
    authorized = sum(step.authorized for step in validated)
    return replace(plan, steps=tuple(validated), status="validated",
                   validation_result={"authorized": authorized,
                                      "unauthorized": len(validated) - authorized},
                   approval_status=("required" if any(
                       step.requires_approval and step.authorized for step in validated)
                       else "not_required"))


def update_step(plan: OrchestratorPlan, step_id: str, **changes) -> OrchestratorPlan:
    return replace(plan, steps=tuple(
        replace(step, **changes) if step.step_id == step_id else step for step in plan.steps))
