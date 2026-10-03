"""Deterministic, side-effect-free task scenario modeling."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from typing import Iterable

import action_item_sentinel
import project_intelligence


SUPPORTED_OPERATIONS = {
    "assign_task", "reassign_task", "change_due_date", "change_priority",
    "complete_task", "leave_unchanged", "workload_redistribution",
}


@dataclass(frozen=True)
class ScenarioRequest:
    operation: str
    goal: str
    task_reference: str | None = None
    assignee_names: tuple[str, ...] = ()
    priority: str | None = None
    due_date: date | None = None
    due_date_offset_days: int | None = None
    target_unassigned: bool = False
    target_priority: str | None = None
    target_overdue: bool = False
    target_owner_name: str | None = None
    target_owner_self: bool = False
    target_due: str | None = None
    selector_plural: bool = False
    target_assignee: str | None = None
    compare: bool = False


@dataclass(frozen=True)
class SimulationResult:
    scenario_id: str
    fingerprint: str
    requester_id: str
    created_at: float
    expires_at: float
    goal: str
    operation: str
    source_task_ids: tuple[str, ...]
    parameters: dict
    baseline_snapshot_version: str
    baseline_task_fingerprints: dict
    baseline_metrics: dict
    simulated_metrics: dict
    impact: dict
    positive: tuple[str, ...]
    tradeoffs: tuple[str, ...]
    unchanged: tuple[str, ...]
    assumptions: tuple[str, ...]
    confidence: str = "deterministic"
    status: str = "simulated"
    decision_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_reference(value: str) -> str:
    value = re.sub(r"\b(?:the|this|that)\s+task\b", " ", value, flags=re.I)
    value = re.sub(r"\btask\b", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip(" .?!\"")


def _semantic_selector(value: str) -> dict:
    """Extract factual task filters while retaining genuine title text."""
    raw = re.sub(r"\s+", " ", str(value or "")).strip()
    lower = raw.casefold()
    priority = re.search(r"\b(P[1-4])\b", raw, re.I)
    owner = re.search(r"\b([A-Za-z][\w.-]*)['’]s\s+(?:P[1-4]\s+)?tasks?\b", raw, re.I)
    target_due = None
    if re.search(r"\bdue\s+today\b", lower):
        target_due = "today"
    elif re.search(r"\bdue\s+tomorrow\b", lower):
        target_due = "tomorrow"
    elif re.search(r"\bdue\s+within\s+48\s+hours?\b", lower):
        target_due = "within_48h"
    semantic = bool(
        re.search(r"\bunassigned\b|\boverdue\b|\bmy\s+(?:p[1-4]\s+)?tasks?\b", lower)
        or owner or target_due
    )
    return {
        "task_reference": None if semantic else _clean_reference(raw),
        "target_unassigned": bool(re.search(r"\bunassigned\b", lower)),
        "target_priority": (priority.group(1).upper() if priority else
                            "P1" if re.search(r"\b(?:high|highest|urgent|critical)[ -]priority\b", lower)
                            else None),
        "target_overdue": bool(re.search(r"\boverdue\b", lower)),
        "target_owner_name": owner.group(1) if owner else None,
        "target_owner_self": bool(re.search(r"\bmy\s+(?:p[1-4]\s+)?tasks?\b", lower)),
        "target_due": target_due,
        "selector_plural": bool(re.search(r"\b(?:all|every|tasks)\b", lower)),
    }


def _next_weekday(today: date, weekday: int) -> date:
    days = (weekday - today.weekday()) % 7
    return today + timedelta(days=days or 7)


def _scenario_date(value: str, today: date) -> date | None:
    lower = value.casefold().strip(" .?!")
    if lower == "today":
        return today
    if lower == "tomorrow":
        return today + timedelta(days=1)
    weekdays = {name: index for index, name in enumerate(
        ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"))}
    for name, weekday in weekdays.items():
        if re.fullmatch(rf"(?:next\s+)?{name}", lower):
            return _next_weekday(today, weekday)
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def parse_request(text: str, today: date | None = None) -> dict | None:
    """Parse explicit simulation and ledger language without an LLM."""
    today = today or date.today()
    raw = re.sub(r"\s+", " ", str(text or "")).strip().rstrip(".?!")
    lower = raw.casefold().rstrip(".?!")

    decision = re.fullmatch(r"(?:what happened after|show)\s+decision\s+([a-f0-9]{8,16})", lower)
    if decision:
        return {"intent": "simulation", "simulation_mode": "show_decision",
                "decision_id": decision.group(1)}
    expected = re.fullmatch(
        r"compare expected (?:vs|versus) actual(?: for)? decision\s+([a-f0-9]{8,16})", lower)
    if expected:
        return {"intent": "simulation", "simulation_mode": "verify_decision",
                "decision_id": expected.group(1)}
    show_one = re.fullmatch(r"show simulation\s+([a-f0-9]{8,16})", lower)
    if show_one:
        return {"intent": "simulation", "simulation_mode": "show_scenario",
                "scenario_id": show_one.group(1)}
    if re.fullmatch(r"(?:show\s+)?(?:my\s+)?(?:recent simulations|decision history|failed decisions)", lower):
        return {"intent": "simulation", "simulation_mode": "history",
                "failed_only": "failed" in lower}
    if re.fullmatch(r"(?:why did we choose this|what happened after that decision)", lower):
        return {"intent": "simulation", "simulation_mode": "show_decision"}
    if re.fullmatch(r"compare expected (?:vs|versus) actual", lower):
        return {"intent": "simulation", "simulation_mode": "verify_decision"}
    if re.fullmatch(r"(?:use|prepare)\s+(?:this|that|the current)\s+scenario", lower):
        return {"intent": "simulation", "simulation_mode": "prepare"}

    explicit = bool(re.match(
        r"^(?:what (?:would )?happens? if|what if|simulate|model|compare|what would change|"
        r"what would improve|what would reduce|hypothetical(?:ly)?|"
        r"if (?:i|we) (?:move|moved|change|changed))", lower))
    if not explicit:
        return None
    if re.search(r"\b(?:do nothing|nothing changes|leave (?:everything|the current .+?) (?:unchanged|as it is))\b", lower):
        request = ScenarioRequest("leave_unchanged", raw)
        return {"intent": "simulation", "simulation_mode": "create", "scenario": asdict(request)}
    if re.search(r"(?:safest way to reduce|what would reduce) (?:our |the )?(?:current )?deadline risk", lower):
        request = ScenarioRequest("workload_redistribution", raw)
        return {"intent": "simulation", "simulation_mode": "create", "scenario": asdict(request)}

    compare = re.search(
        r"compare(?: the impact of)? assigning (.+?) to ([A-Za-z][\w.-]*)\s+(?:vs|versus|or)\s+([A-Za-z][\w.-]*)$",
        raw, re.I)
    if compare:
        selector = _semantic_selector(compare.group(1))
        request = ScenarioRequest(
            "assign_task", raw, assignee_names=(compare.group(2), compare.group(3)),
            target_assignee=compare.group(2), compare=True, **selector)
        return {"intent": "simulation", "simulation_mode": "compare", "scenario": asdict(request)}

    assign = re.search(
        r"(?:assign(?:ing)?|give|giving)\s+(.+?)\s+to\s+([A-Za-z][\w.-]*)$", raw, re.I)
    if assign:
        selector = _semantic_selector(assign.group(1))
        request = ScenarioRequest(
            "assign_task", raw, assignee_names=(assign.group(2),),
            target_assignee=assign.group(2), **selector)
        return {"intent": "simulation", "simulation_mode": "create", "scenario": asdict(request)}

    due = re.search(
        r"(?:move|moving|moved)\s+(.+?)\s+(?:(?:deadline|due date)\s+)?to\s+"
        r"(.+?)(?:\s*,?\s*what would happen)?$", raw, re.I)
    if not due:
        due = re.search(
            r"(.+?)\s+(?:are|is)\s+moved\s+to\s+(.+?)(?:\s*,?\s*what would happen)?$",
            raw, re.I)
    if not due:
        due = re.search(r"(?:if\s+)?(.+?)\s+is\s+due\s+(.+)$", raw, re.I)
    if due and (parsed_date := _scenario_date(due.group(2), today)):
        selector = _semantic_selector(due.group(1))
        request = ScenarioRequest(
            "change_due_date", raw, due_date=parsed_date, **selector)
        return {"intent": "simulation", "simulation_mode": "create", "scenario": asdict(request)}

    shifted = re.search(
        r"(?:move|moving|moved)\s+(.+?)\s+by\s+(?:one|1)\s+week"
        r"(?:\s*,?\s*what would happen)?$", raw, re.I)
    if shifted:
        selector = _semantic_selector(shifted.group(1))
        request = ScenarioRequest(
            "change_due_date", raw, due_date_offset_days=7, **selector)
        return {"intent": "simulation", "simulation_mode": "create", "scenario": asdict(request)}

    priority = re.search(r"(?:make|making)\s+(.+?)\s+(P[1-4])$", raw, re.I)
    if priority:
        selector = _semantic_selector(priority.group(1))
        request = ScenarioRequest(
            "change_priority", raw, priority=priority.group(2).upper(), **selector)
        return {"intent": "simulation", "simulation_mode": "create", "scenario": asdict(request)}
    complete = re.search(r"(?:complete|completing|finish|finishing)\s+(.+)$", raw, re.I)
    if complete:
        selector = _semantic_selector(complete.group(1))
        request = ScenarioRequest("complete_task", raw, **selector)
        return {"intent": "simulation", "simulation_mode": "create", "scenario": asdict(request)}
    if lower.startswith("compare"):
        # Existing progress/status comparisons own non-scenario comparison
        # language; simulation only intercepts the explicit assignment form.
        return None
    return {"intent": "clarify", "clarification": (
        "Please specify the task and hypothetical assignment, deadline, priority, or completion change.")}


def snapshot_version(tasks: Iterable[project_intelligence.NormalizedTask]) -> str:
    state = [{
        "id": task.item_id, "owners": task.owner_ids, "priority": task.priority,
        "due": task.due_date.isoformat() if task.due_date else None,
        "completed": task.completed, "name": task.name,
    } for task in tasks]
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def task_state_fingerprint(task: project_intelligence.NormalizedTask) -> str:
    return hashlib.sha256(json.dumps({
        "id": task.item_id, "owners": task.owner_ids, "priority": task.priority,
        "due": task.due_date.isoformat() if task.due_date else None,
        "completed": task.completed, "name": task.name,
    }, sort_keys=True).encode()).hexdigest()


def calculate_metrics(tasks, today: date) -> dict:
    pending = [task for task in tasks if not task.completed]
    owners = {}
    for task in pending:
        for owner in task.owner_ids:
            row = owners.setdefault(owner, {"pending": 0, "p1": 0, "overdue": 0, "due_soon": 0})
            row["pending"] += 1
            row["p1"] += task.priority == "P1"
            row["overdue"] += bool(task.due_date and task.due_date < today)
            row["due_soon"] += bool(task.due_date and today <= task.due_date <= today + timedelta(days=2))
    priorities = {key: sum(task.priority == key for task in pending) for key in ("P1", "P2", "P3", "P4")}
    return {
        "pending": len(pending), "completed": len(tasks) - len(pending),
        "unassigned": sum(not task.owner_ids for task in pending),
        "overdue": sum(bool(task.due_date and task.due_date < today) for task in pending),
        "due_today": sum(task.due_date == today for task in pending),
        "due_24h": sum(bool(task.due_date and today <= task.due_date <= today + timedelta(days=1)) for task in pending),
        "due_48h": sum(bool(task.due_date and today <= task.due_date <= today + timedelta(days=2)) for task in pending),
        "due_this_week": sum(bool(task.due_date and today <= task.due_date <= today + timedelta(days=7)) for task in pending),
        "priorities": priorities, "owners": owners,
    }


def _risk_keys(tasks, today):
    settings = action_item_sentinel.SentinelSettings.from_env()
    risks = action_item_sentinel.detect_risks(
        tasks, today, settings.warning_days, settings.combined_task_threshold)
    return {f"{risk.risk_type}:{','.join(sorted(risk.task_ids))}" for risk in risks}


def project(tasks, operation: str, task_ids: Iterable[str], parameters: dict):
    """Clone normalized tasks and apply a hypothetical change only to the clone."""
    wanted = set(task_ids)
    projected = []
    for task in tasks:
        clone = replace(task)
        if task.item_id in wanted:
            if operation in {"assign_task", "reassign_task", "workload_redistribution"}:
                clone = replace(clone, owner_ids=tuple(parameters.get("assignee_ids") or ()))
            elif operation == "change_due_date":
                if parameters.get("due_date_offset_days") is not None:
                    clone = replace(
                        clone, due_date=(clone.due_date + timedelta(
                            days=int(parameters["due_date_offset_days"])))
                        if clone.due_date else None)
                else:
                    clone = replace(clone, due_date=date.fromisoformat(parameters["due_date"]))
            elif operation == "change_priority":
                clone = replace(clone, priority=parameters["priority"])
            elif operation == "complete_task":
                clone = replace(clone, completed=True, status="Completed")
        projected.append(clone)
    return projected


def simulate(*, requester_id: str, goal: str, operation: str, tasks,
             task_ids: Iterable[str], parameters: dict, now: float | None = None,
             today: date | None = None, ttl_seconds: int = 1800) -> SimulationResult:
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError("That simulation operation is not supported.")
    now, today = (time.time() if now is None else now), (today or date.today())
    tasks, task_ids = list(tasks), tuple(task_ids)
    projected = project(tasks, operation, task_ids, parameters)
    baseline, modeled = calculate_metrics(tasks, today), calculate_metrics(projected, today)
    baseline_risks, modeled_risks = _risk_keys(tasks, today), _risk_keys(projected, today)
    impact = {
        "pending_delta": modeled["pending"] - baseline["pending"],
        "completed_delta": modeled["completed"] - baseline["completed"],
        "unassigned_delta": modeled["unassigned"] - baseline["unassigned"],
        "overdue_delta": modeled["overdue"] - baseline["overdue"],
        "risk_removed": sorted(baseline_risks - modeled_risks),
        "risk_introduced": sorted(modeled_risks - baseline_risks),
        "risk_unchanged": sorted(baseline_risks & modeled_risks),
    }
    owner_id = next(iter(parameters.get("assignee_ids") or ()), None)
    before_owner = baseline["owners"].get(owner_id, {}) if owner_id else {}
    after_owner = modeled["owners"].get(owner_id, {}) if owner_id else {}
    impact["owner_delta"] = {
        key: after_owner.get(key, 0) - before_owner.get(key, 0)
        for key in ("pending", "p1", "overdue", "due_soon")
    }
    positive, negative, unchanged = [], [], []
    if impact["unassigned_delta"] < 0:
        positive.append(f"Removes {-impact['unassigned_delta']} unassigned task" +
                        ("s" if impact["unassigned_delta"] != -1 else ""))
    if impact["completed_delta"] > 0:
        positive.append(f"Models {impact['completed_delta']} additional completed task" +
                        ("s" if impact["completed_delta"] != 1 else ""))
    if impact["overdue_delta"] < 0:
        positive.append(f"Reduces overdue work by {-impact['overdue_delta']}")
    if impact["risk_removed"]:
        positive.append(f"Removes {len(impact['risk_removed'])} current risk signal" +
                        ("s" if len(impact["risk_removed"]) != 1 else ""))
    if impact["owner_delta"].get("pending", 0) > 0:
        negative.append("The selected owner's pending workload increases")
    if impact["owner_delta"].get("p1", 0) > 0:
        negative.append("The selected owner's P1 concentration increases")
    if impact["risk_introduced"]:
        negative.append(f"Introduces {len(impact['risk_introduced'])} modeled risk signal" +
                        ("s" if len(impact["risk_introduced"]) != 1 else ""))
    for key, label in (("overdue_delta", "Overdue count"), ("pending_delta", "Total pending count")):
        if impact[key] == 0:
            unchanged.append(f"{label} remains unchanged")
    if operation == "leave_unchanged":
        unchanged = ["Current unresolved exposure remains unchanged"]
    baseline_version = snapshot_version(tasks)
    seed = json.dumps({"requester": requester_id, "operation": operation,
                       "tasks": task_ids, "parameters": parameters,
                       "baseline": baseline_version}, sort_keys=True)
    fingerprint = hashlib.sha256(seed.encode()).hexdigest()
    return SimulationResult(
        scenario_id=fingerprint[:10], fingerprint=fingerprint,
        requester_id=requester_id, created_at=now, expires_at=now + ttl_seconds,
        goal=goal, operation=operation, source_task_ids=task_ids,
        parameters=parameters, baseline_snapshot_version=baseline_version,
        baseline_task_fingerprints={task.item_id: task_state_fingerprint(task)
                                    for task in tasks if task.item_id in set(task_ids)},
        baseline_metrics=baseline, simulated_metrics=modeled, impact=impact,
        positive=tuple(positive), tradeoffs=tuple(negative), unchanged=tuple(unchanged),
        assumptions=("Only the selected fields change; all other task state remains constant.",
                     "Metrics are projected from the current authorized Slack List snapshot."),
    )


def is_stale(result: SimulationResult, current_tasks) -> bool:
    current = {task.item_id: task for task in current_tasks}
    return any(task_id not in current or task_state_fingerprint(current[task_id]) != fingerprint
               for task_id, fingerprint in result.baseline_task_fingerprints.items())
