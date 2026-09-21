"""Validated mutation plans and exact-ID read-after-write verification."""
from dataclasses import dataclass, field
from datetime import date

import config
import slack_tools
import delivery


@dataclass
class MutationResult:
    item_id: str
    item: dict | None = None
    verified: bool = False
    problems: list = field(default_factory=list)
    outcome: str = "unable_to_verify"


def authorize_collection(item_ids, current_items, intent, changes, ctx, schema):
    """Authorize and validate the complete exact-ID target set before writes."""
    if not config.has_permission(ctx, intent):
        raise PermissionError(f"You do not have permission to {intent} action items.")
    if len(item_ids) > 1 and not config.has_permission(ctx, "bulk"):
        raise PermissionError("Your role cannot perform bulk action-item operations.")
    live_ids = {slack_tools.extract_item_id(item) for item in current_items}
    if any(item_id not in live_ids for item_id in item_ids):
        raise ValueError("One or more selected action items no longer exist in the Slack List; no changes were made.")
    by_id = {slack_tools.extract_item_id(item): item for item in current_items}
    if not config.has_permission(ctx, "update_others") and any(
            ctx.user_id not in slack_tools.extract_assignee_ids(by_id[item_id], schema) for item_id in item_ids):
        raise PermissionError("Your role can only modify tasks assigned to you.")
    for change in changes:
        if change["field"] == "completed":
            allowed = config.has_permission(ctx, intent if intent in {"complete", "reopen"} else "complete")
        else:
            allowed = config.can_edit_field(ctx, change["field"])
        if not allowed:
            raise PermissionError(f"Your role cannot edit the {change['field']} field.")
        if change["field"] == "assignee":
            required = {"reassign" if slack_tools.extract_assignee_ids(by_id[item_id], schema) else "assign"
                        for item_id in item_ids}
            missing = [permission for permission in required if not config.has_permission(ctx, permission)]
            if missing:
                raise PermissionError(f"Your role cannot {missing[0]} action items.")
    return tuple(item_ids)


def prepare_changes(parsed, ctx, schema, today):
    intent = parsed["intent"]
    if not config.has_permission(ctx, intent):
        raise PermissionError(f"You do not have permission to {intent} action items.")
    if intent == "delete":
        return []
    if intent in {"complete", "reopen"}:
        changes = [{"field": "completed", "value": intent == "complete"}]
    else:
        changes = parsed.get("changes") or []
        if not changes and parsed.get("field"):
            changes = [{"field": parsed["field"], "value": parsed.get("new_name") or parsed.get("value")}]
        if not changes:
            raise ValueError("Please specify what should change and the new value.")
    normalized = []
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("Each change must specify a field and value.")
        name, value = change.get("field"), change.get("value")
        name = {"due": "due_date", "date": "due_date", "deadline": "due_date",
                "title": "name", "task": "name", "owner": "assignee"}.get(name, name)
        allowed = (config.has_permission(ctx, intent if intent in {"complete", "reopen"} else "complete")
                   if name == "completed" else config.can_edit_field(ctx, name))
        if not allowed:
            raise PermissionError(f"Your role cannot edit the {name} field.")
        if name == "due_date":
            value = date.fromisoformat(str(value)).isoformat()
            if value < today.isoformat():
                raise ValueError("The due date cannot be in the past.")
        elif name == "priority":
            value = config.normalize_priority(value)
            if not value:
                raise ValueError("Priority must be P1, P2, P3 or P4.")
        elif name == "assignee":
            values = value if isinstance(value, list) else [value]
            resolved = []
            for candidate in values:
                user_id = ctx.user_id if str(candidate).casefold() in {"me", "myself"} else slack_tools.find_user_id(candidate)
                if not user_id:
                    raise ValueError(f"I couldn't resolve the new assignee {candidate!r}.")
                resolved.append(user_id)
            value = list(dict.fromkeys(resolved))
            if len(value) == 1:
                value = value[0]
            if not value:
                raise ValueError("I couldn't resolve the new assignee.")
        elif name == "name":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Task names cannot be empty.")
            value = value.strip()
        elif name == "status":
            value = config.normalize_status(value)
            if not value:
                raise ValueError("Please specify a valid task status.")
            # Completion and open share the canonical checkbox state.
            if value in {"open", "completed"}:
                name, value = "completed", value == "completed"
        if name == "completed" and not isinstance(value, bool):
            raise ValueError("Completed must be a boolean.")
        # Validate every cell before the first write, avoiding preventable partial updates.
        slack_tools._write_cell(schema, name, value)
        normalized.append({"field": name, "value": value})
    return normalized


def verify(item_id, changes, ctx, schema, deleted=False):
    result = MutationResult(item_id)
    try:
        items = slack_tools.list_action_items(ctx, ctx.list_id)
    except Exception:
        result.problems.append("verification read failed; outcome is unknown")
        return result
    result.item = next((x for x in items if slack_tools.extract_item_id(x) == item_id), None)
    if deleted:
        if result.item:
            result.problems.append("task still exists in Slack List")
    elif result.item is None:
        result.problems.append("selected task was not found during verification")
    else:
        extractors = {
            "name": slack_tools.extract_item_name, "priority": slack_tools.extract_priority,
            "assignee": slack_tools.extract_assignee_ids, "due_date": slack_tools.extract_due_date,
            "completed": slack_tools.extract_completed, "status": slack_tools.extract_status,
        }
        for change in changes:
            name, expected = change["field"], change["value"]
            extractor = extractors.get(name)
            actual = extractor(result.item, schema) if extractor else slack_tools.extract_field_value(result.item, schema, name)
            if name == "assignee":
                expected = expected if isinstance(expected, list) else [expected]
            if actual != expected:
                problem = "task status in Slack List still shows pending" if name == "completed" and expected is True else f"{name}: expected {expected!r}, found {actual!r}"
                result.problems.append(problem)
    result.verified = not result.problems
    result.outcome = "verified_success" if result.verified else (
        "partial_success" if result.item and 0 < len(result.problems) < len(changes) else "failed")
    return result


def execute(item_id, intent, changes, ctx, schema):
    key = delivery.checkpoint_key("mutation", {"list": ctx.list_id, "id": item_id, "intent": intent, "changes": changes})
    prior = delivery.checkpoint_read(key)
    if prior:
        observed = verify(item_id, changes, ctx, schema, deleted=intent == "delete")
        if observed.verified or prior["status"] == "verified":
            # Do not undo a later external edit when replaying a completed operation.
            return observed
    delivery.checkpoint_write(key, "started", {"item_id": item_id})
    write_error = False
    try:
        if intent == "delete":
            slack_tools.delete_action_item(item_id, ctx, ctx.list_id)
        elif intent == "complete":
            slack_tools.complete_action_item(item_id, ctx, ctx.list_id)
        elif intent == "reopen":
            slack_tools.reopen_action_item(item_id, ctx, ctx.list_id)
        else:
            for change in changes:
                slack_tools.update_action_item_field(item_id, change["field"], change["value"], ctx, ctx.list_id)
    except Exception:
        # A timeout can happen after Slack committed a write. Read back even on errors.
        write_error = True
    result = verify(item_id, changes, ctx, schema, deleted=intent == "delete")
    if write_error and not result.verified:
        result.problems.insert(0, "write failed or was interrupted; some fields may have changed")
    delivery.checkpoint_write(key, "verified" if result.verified else "unverified", {"item_id": item_id})
    return result


def execute_collection(item_ids, intent, changes, ctx, schema):
    """Execute and independently verify a pre-resolved exact-ID collection."""
    return [execute(item_id, intent, changes, ctx, schema) for item_id in item_ids]
