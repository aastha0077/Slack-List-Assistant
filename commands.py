"""Validate untrusted parser/model output before it reaches application services."""
from copy import deepcopy

INTENTS = {"create", "list", "inspect", "source", "progress", "focus", "weekly_focus", "health", "sentinel", "command_center", "intelligence_summary", "orchestrator", "simulation", "visual_analytics", "similar_tasks", "plan", "workload", "standup", "weekly_summary",
           "apply_proposal", "confirm", "cancel", "history", "dependencies",
           "update", "complete", "reopen", "delete", "members", "compound",
           "clarify", "out_of_scope", "temporarily_unavailable"}
SORT_FIELDS = {"name", "due_date", "priority", "status", "assignee"}
GROUP_FIELDS = {"assignee", "status", "priority", "due_date"}
AGGREGATIONS = {"count"}
SELECTION_MODES = {"one", "many", "collection"}
SELECTION_ORDER_FIELDS = {"position", "created_at", "due_date", "priority", "name", "status", "assignee", "urgency"}
RESULT_OPERATIONS = {"return_collection", "select_one", "select_many", "aggregate", "comparison", "summary"}
TEMPORAL_FIELDS = {"due_date", "completed_at", "created_at", "updated_at"}
TEMPORAL_RELATIONS = {"on", "before", "after", "between"}
ANALYTICS_METRICS = {
    "overview", "completion", "workload", "status_distribution", "priority_distribution",
    "overdue", "due_today", "due_this_week", "upcoming", "at_risk",
    "completed_over_time", "created_over_time", "comparison", "summary",
}


def validate_command(value):
    if not isinstance(value, dict) or value.get("intent") not in INTENTS:
        raise ValueError("Please specify a Slack List action and its target.")
    result = deepcopy(value)
    # Only the resolver may attach exact IDs. The language layer has no authority to do so.
    result.pop("target_ids", None)
    result.pop("resolved_assignee_ids", None)
    result.pop("resolved_member_ids", None)
    result.pop("actor_id", None)
    if result.get("sentinel_mode") not in {None, "risks", "alerts", "explain", "action"}:
        raise ValueError("Please specify a supported Sentinel view.")
    if result.get("sentinel_action") not in {None, "approve", "dismiss"}:
        raise ValueError("Please specify a supported Sentinel action.")
    if result.get("command_center_mode") not in {
            None, "overview", "owner_risk", "risk_followup", "prepare_message"}:
        raise ValueError("Please specify a supported Command Center view.")
    if result.get("intelligence_mode") not in {
            None, "summary", "emerging_risks", "deadline_pressure", "workload_outlook"}:
        raise ValueError("Please specify a supported intelligence view.")
    if result.get("orchestrator_mode") not in {
            None, "create", "approve", "cancel", "explain", "remove_step", "show_context"}:
        raise ValueError("Please specify a supported orchestration operation.")
    if result.get("simulation_mode") not in {
            None, "create", "compare", "prepare", "history", "show_scenario",
            "show_decision", "verify_decision"}:
        raise ValueError("Please specify a supported simulation operation.")
    if result.get("response_mode") not in {None, "text", "chart", "dashboard", "table"}:
        raise ValueError("Please specify a supported response mode.")
    if result.get("visualization_type") not in {
            None, "workload", "priority", "completion", "deadlines",
            "completed_trend", "created_trend", "all_tasks", "overdue_tasks", "upcoming_tasks",
            "dashboard"}:
        raise ValueError("Please specify a supported visualization type.")
    if result.get("chart_type") not in {
            None, "auto", "bar", "pie", "line", "table", "dashboard"}:
        raise ValueError("Please specify a supported chart type.")
    operations = result.get("operations") or []
    if not isinstance(operations, list):
        raise ValueError("Please provide valid action-item operations.")
    if result["intent"] == "compound":
        if len(operations) < 2:
            raise ValueError("A compound request needs at least two operations.")
        normalized_operations = []
        for operation in operations:
            if not isinstance(operation, dict) or operation.get("intent") == "compound":
                raise ValueError("Nested compound operations are not supported.")
            normalized_operations.append(validate_command(operation))
        result["operations"] = normalized_operations
    elif operations:
        raise ValueError("Multiple operations require a compound request.")
    for name in ("task_name", "assignee", "member", "role", "priority", "due_date", "status", "query", "dependency_origin",
                 "date_from", "date_to", "sentinel_mode", "sentinel_action", "command_center_mode", "intelligence_mode",
                 "response_mode", "visualization_type", "orchestrator_mode", "simulation_mode",
                 "plan_id", "step_id", "goal", "decision_id", "scenario_id"):
        if result.get(name) is not None and not isinstance(result[name], str):
            raise ValueError(f"Please provide a valid {name.replace('_', ' ')}.")
    if result.get("chart_type") is not None and not isinstance(result["chart_type"], str):
        raise ValueError("Please provide a valid chart type.")
    if result.get("step_number") is not None and (
            not isinstance(result["step_number"], int) or result["step_number"] < 1):
        raise ValueError("Please provide a valid plan step number.")
    statuses = result.get("statuses")
    if statuses is not None:
        if (not isinstance(statuses, list)
                or any(value not in {"open", "completed"} for value in statuses)):
            raise ValueError("Please specify valid task statuses.")
        result["statuses"] = list(dict.fromkeys(statuses))
    for name, allowed in (("sort_by", SORT_FIELDS), ("group_by", GROUP_FIELDS),
                          ("aggregate", AGGREGATIONS)):
        if result.get(name) is not None and result[name] not in allowed:
            raise ValueError(f"Please specify a supported {name.replace('_', ' ')}.")
    if result.get("sort_order") not in {None, "asc", "desc"}:
        raise ValueError("Please specify ascending or descending sort order.")
    if result.get("result_operation") not in {None, *RESULT_OPERATIONS}:
        raise ValueError("Please specify a supported result operation.")
    temporal = result.get("temporal_filter")
    if temporal is not None:
        if not isinstance(temporal, dict) or temporal.get("field") not in TEMPORAL_FIELDS:
            raise ValueError("Please specify which task date or time should be evaluated.")
        if temporal.get("relation") not in TEMPORAL_RELATIONS:
            raise ValueError("Please specify a supported temporal comparison.")
        for key in ("date", "date_from", "date_to"):
            if temporal.get(key) is not None and not isinstance(temporal[key], str):
                raise ValueError("Please specify a valid temporal date.")
    target_selection = result.get("target_selection")
    if target_selection is not None:
        if not isinstance(target_selection, dict) or target_selection.get("mode") not in SELECTION_MODES:
            raise ValueError("Please clarify whether you want one task, several tasks, or the collection.")
        if target_selection.get("order_by") not in {None, *SELECTION_ORDER_FIELDS}:
            raise ValueError("Please specify a supported task-selection order.")
        if target_selection.get("direction") not in {None, "asc", "desc"}:
            raise ValueError("Please specify a valid task-selection direction.")
        if target_selection.get("count") is not None and (type(target_selection["count"]) is not int or target_selection["count"] < 1):
            raise ValueError("Please specify a valid number of tasks to select.")
        if target_selection["mode"] in {"one", "many"} and not target_selection.get("order_by"):
            raise ValueError("Please specify how the requested task selection should be made.")
    if result.get("assignees") is not None:
        if not isinstance(result["assignees"], list) or any(not isinstance(x, str) or not x.strip() for x in result["assignees"]):
            raise ValueError("Please provide valid assignee references.")
        result["assignees"] = list(dict.fromkeys(x.strip() for x in result["assignees"]))
    if result.get("assignee") and not result.get("assignees"):
        result["assignees"] = [result["assignee"]]
    if result.get("members") is not None:
        if not isinstance(result["members"], list) or any(not isinstance(x, str) or not x.strip() for x in result["members"]):
            raise ValueError("Please provide valid workspace-member references.")
        result["members"] = list(dict.fromkeys(x.strip() for x in result["members"]))
    if result.get("member") and not result.get("members"):
        result["members"] = [result["member"]]
    if result.get("assignee_reference") not in {None, "context"}:
        raise ValueError("Please clarify which users you mean.")
    if result.get("assignee_condition") not in {None, "self", "other", "unassigned", "assigned"}:
        raise ValueError("Please specify a supported assignee condition.")
    metrics = result.get("analytics_metrics") or []
    if (not isinstance(metrics, list) or any(metric not in ANALYTICS_METRICS for metric in metrics)):
        raise ValueError("Please specify supported progress metrics.")
    result["analytics_metrics"] = list(dict.fromkeys(metrics))
    period = result.get("analytics_period")
    if period is not None:
        if not isinstance(period, dict) or any(key not in {"start", "end"} for key in period):
            raise ValueError("Please specify a valid progress period.")
        if any(value is not None and not isinstance(value, str) for value in period.values()):
            raise ValueError("Please specify valid progress period dates.")
    comparison = result.get("analytics_comparison")
    if comparison is not None:
        if not isinstance(comparison, dict) or set(comparison) != {"current", "previous"}:
            raise ValueError("Please specify two valid analytics comparison periods.")
        for value in comparison.values():
            if not isinstance(value, dict) or set(value) != {"start", "end"} or any(
                    not isinstance(value[key], str) for key in ("start", "end")):
                raise ValueError("Please specify valid analytics comparison dates.")
    planning_period = result.get("planning_period")
    if planning_period is not None:
        if not isinstance(planning_period, dict) or set(planning_period) != {"start", "end"}:
            raise ValueError("Please specify a valid planning period.")
        if any(not isinstance(planning_period[key], str) for key in ("start", "end")):
            raise ValueError("Please specify valid planning dates.")
    if result.get("target_scope") not in {None, "single", "multiple", "filtered", "all_applicable", "contextual"}:
        raise ValueError("Please clarify the intended task collection.")
    for name in ("assignee_self", "member_self", "all_tasks", "due_today", "overdue", "due_this_week",
                 "literal_name", "count_only", "attention_only", "risk_view", "health_summary",
                 "focus_intelligence", "recommend_balance", "explicit_visual"):
        if name in result and not isinstance(result[name], bool):
            raise ValueError(f"Invalid {name} flag in the interpreted request.")
    if result.get("completed") is not None and not isinstance(result["completed"], bool):
        raise ValueError("Please clarify whether you want pending or completed tasks.")
    for name in ("limit", "selection_index", "selection_count"):
        if result.get(name) is not None and type(result[name]) is not int:
            raise ValueError("Please specify a valid task number or count.")
    if result.get("selection_numbers") is not None:
        if not isinstance(result["selection_numbers"], list) or any(type(x) is not int for x in result["selection_numbers"]):
            raise ValueError("Please specify valid displayed task numbers.")
    reference = result.get("reference")
    if reference is not None:
        if not isinstance(reference, dict) or reference.get("kind") not in {"positions", "focus", "previous", "focus_set", "relative", "all", "both", "head", "tail"}:
            raise ValueError("Please clarify which displayed task you mean.")
        if not isinstance(reference.get("positions", []), (list, tuple)) or any(type(x) is not int for x in reference.get("positions", [])):
            raise ValueError("Please specify valid displayed positions.")
        if type(reference.get("count", 0)) is not int:
            raise ValueError("Please specify a valid task count.")
    changes = result.get("changes") or []
    if not isinstance(changes, list):
        raise ValueError("Please specify valid field changes.")
    for change in changes:
        if (not isinstance(change, dict) or not isinstance(change.get("field"), str)
                or not change["field"].strip() or "value" not in change):
            raise ValueError("Please specify a supported field and its new value.")
    tasks = result.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError("Please provide a list of task requests.")
    normalized = []
    for task in tasks:
        if isinstance(task, str):
            task = {"task_name": task}
        if not isinstance(task, dict):
            raise ValueError("Each task needs a name or reference.")
        task = dict(task)
        source = task.get("_source")
        if source is not None:
            if (not isinstance(source, dict)
                    or any(key not in {"type", "reference", "confidence", "evidence"} for key in source)
                    or not isinstance(source.get("type"), str)
                    or source.get("reference") is not None and not isinstance(source.get("reference"), str)
                    or source.get("evidence") is not None and not isinstance(source.get("evidence"), str)
                    or source.get("confidence") is not None and not isinstance(source.get("confidence"), (int, float))):
                raise ValueError("Please provide valid action-item source metadata.")
            task["_source"] = {
                "type": source["type"][:32],
                "reference": (source.get("reference") or "")[:200],
                "confidence": max(0.0, min(1.0, float(source.get("confidence", 0.0)))),
                "evidence": (source.get("evidence") or "")[:240],
            }
        task["intent"] = result["intent"]
        if task.get("tasks"):
            raise ValueError("Nested task batches are not supported.")
        normalized.append(validate_command(task))
    result["tasks"] = normalized
    groups = result.get("target_groups") or []
    if not isinstance(groups, list):
        raise ValueError("Please provide valid task target groups.")
    normalized_groups = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("query"), str) or not group["query"].strip():
            raise ValueError("Each task target group needs a search constraint.")
        clean = {"query": group["query"].strip()}
        reference = group.get("reference")
        if reference is not None:
            if (not isinstance(reference, dict)
                    or reference.get("kind") not in {"positions", "focus", "previous", "focus_set", "relative", "all", "both", "head", "tail"}
                    or any(type(x) is not int for x in reference.get("positions", []))
                    or type(reference.get("count", 0)) is not int):
                raise ValueError("Please specify a valid selection for each task group.")
            clean["reference"] = reference
        normalized_groups.append(clean)
    result["target_groups"] = normalized_groups
    return result
