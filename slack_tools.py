
import json
import logging
import os
import re
from datetime import date
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv
from slack_sdk import WebClient

import config

load_dotenv()
logger = logging.getLogger(__name__)
_client = None

NAME_KEYS = {"name", "title", "task", "task name"}
PRIORITY_KEYS = {"priority"}
ASSIGNEE_KEYS = {"todo_assignee", "assignee", "owner"}
DUE_KEYS = {"todo_due_date", "due_date", "date"}
COMPLETED_KEYS = {"todo_completed", "completed"}
STATUS_KEYS = {"status", "state"}


def configure(client=None):
    global _client
    if client is not None:
        _client = client
    elif _client is None:
        token = os.getenv("SLACK_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("SLACK_BOT_TOKEN is missing")
        _client = WebClient(token=token)
    return _client


def client():
    return _client or configure()


def data_of(response):
    if isinstance(response, dict):
        return response
    data = getattr(response, "data", None)
    return data if isinstance(data, dict) else dict(response)


def checked(response, action="Slack API request"):
    data = data_of(response)
    if not data.get("ok", True):
        raise RuntimeError(
            f"{action} failed: {data.get('error', 'unknown_error')}"
        )
    return data


def _schema_fields(schema):
    if isinstance(schema, dict):
        return schema.get("schema") or schema.get("columns") or []
    return schema if isinstance(schema, list) else []


def column(schema, *, keys=(), names=(), types=()):
    keys = {str(x).casefold() for x in keys}
    names = {str(x).casefold() for x in names}
    types = {str(x).casefold() for x in types}

    for field in _schema_fields(schema):
        if not isinstance(field, dict):
            continue

        fkey = str(field.get("key", "")).casefold()
        fname = str(field.get("name", "")).casefold()
        ftype = str(field.get("type", "")).casefold()

        if fkey in keys or fname in names or ftype in types:
            return field

    return None


def column_id(field):
    return (field or {}).get("id") or (field or {}).get("column_id")


def get_list_schema(list_id: str):
    if not list_id:
        raise RuntimeError("No Slack List is mapped to this channel")

    response = client().slackLists_items_list(
        list_id=list_id,
        limit=1,
        include_list=True,
    )

    data = checked(response, "Read Slack List schema")
    parent = data.get("list") or {}

    metadata = (
        parent.get("list_metadata")
        or data.get("list_metadata")
        or parent
    )

    schema = (
        metadata.get("schema")
        if isinstance(metadata, dict)
        else None
    )

    if not isinstance(schema, list):
        schema = data.get("schema")

    if not isinstance(schema, list):
        raise RuntimeError("Slack did not return a List schema")

    return {
        "list_id": list_id,
        "title": parent.get("title"),
        "schema": schema,
    }


def list_action_items(context=None, list_id=None):
    actual = (
        list_id
        or getattr(context, "list_id", None)
        or config.DEFAULT_LIST_ID
    )

    if not actual:
        raise RuntimeError("No Slack List is configured")

    items = []
    cursor = None

    while True:
        kwargs = {
            "list_id": actual,
            "limit": 100,
            "include_list": False,
        }

        if cursor:
            kwargs["cursor"] = cursor

        data = checked(
            client().slackLists_items_list(**kwargs),
            "Read Slack List items",
        )

        items.extend(data.get("items") or [])

        cursor = (
            data.get("response_metadata") or {}
        ).get("next_cursor") or ""

        if not cursor:
            return items


def extract_item_id(item):
    if not isinstance(item, dict):
        return None

    return (
        item.get("id")
        or item.get("item_id")
        or item.get("itemId")
    )


def _rich_text(value):
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "".join(_rich_text(v) for v in value)

    if isinstance(value, dict):
        if value.get("text") is not None:
            return str(value["text"])

        return _rich_text(value.get("elements", []))

    return ""


# FIX: Never match None values between schema and item fields.
def _item_field(item, field):
    if not field or not isinstance(item, dict):
        return None

    wanted = {
        str(value)
        for value in (
            field.get("id"),
            field.get("column_id"),
            field.get("key"),
        )
        if value is not None
    }

    if not wanted:
        return None

    for cell in item.get("fields", []) or []:
        if not isinstance(cell, dict):
            continue

        actual = {
            str(value)
            for value in (
                cell.get("id"),
                cell.get("column_id"),
                cell.get("key"),
            )
            if value is not None
        }

        if wanted.intersection(actual):
            return cell

    return None


# FIX: Extract the actual title from the matching field.
def extract_item_name(item, schema):
    if not isinstance(item, dict):
        return ""

    fields = item.get("fields", []) or []

    name_column = column(
        schema,
        keys=NAME_KEYS,
        names={"Name", "Title", "Task", "Task Name"},
        types={"text", "rich_text"},
    )

    field = _item_field(item, name_column)

    if field:
        candidates = [
            field.get("text"),
            field.get("rich_text"),
            field.get("value"),
        ]

        for value in candidates:
            if value is None:
                continue

            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    result = _rich_text(parsed).strip()

                    if result:
                        return result

                except (ValueError, TypeError):
                    pass

                if value.strip():
                    return value.strip()

            result = _rich_text(value).strip()

            if result:
                return result

    # Fallback: inspect populated text fields only.
    for cell in fields:
        if not isinstance(cell, dict):
            continue

        value = cell.get("text")

        if value is None:
            value = cell.get("rich_text")

        if value is None:
            value = cell.get("value")

        if value is None:
            continue

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                result = _rich_text(parsed).strip()

                if result:
                    return result

            except (ValueError, TypeError):
                pass

            if value.strip():
                return value.strip()

        result = _rich_text(value).strip()

        if result:
            return result

    return ""


def extract_completed(item, schema):
    field = _item_field(
        item,
        column(
            schema,
            keys=COMPLETED_KEYS,
            names={"Completed"},
            types={"todo_completed", "completed", "checkbox"},
        ),
    )

    if not field:
        return False

    if isinstance(field.get("checkbox"), bool):
        return field["checkbox"]

    value = field.get("value")

    if isinstance(value, bool):
        return value

    return str(value or "").strip().casefold() in {
        "true",
        "yes",
        "1",
        "done",
        "completed",
    }


def extract_priority(item, schema):
    field = _item_field(
        item,
        column(
            schema,
            keys=PRIORITY_KEYS,
            names={"Priority"},
            types={"select", "multi_select"},
        ),
    )

    if not field:
        return ""

    selected = (
        field.get("select")
        or field.get("multi_select")
        or field.get("value")
    )

    if isinstance(selected, list):
        selected = selected[0] if selected else None

    choices = (
        (
            column(
                schema,
                keys=PRIORITY_KEYS,
                names={"Priority"},
            )
            or {}
        ).get("options")
        or {}
    ).get("choices") or []

    for choice in choices:
        if isinstance(choice, dict) and selected in {
            choice.get("id"),
            choice.get("value"),
        }:
            return (
                config.normalize_priority(
                    choice.get("label") or choice.get("name")
                )
                or ""
            )

    return config.normalize_priority(selected) or ""


def extract_assignee_id(item, schema):
    field = _item_field(
        item,
        column(
            schema,
            keys=ASSIGNEE_KEYS,
            names={"Assignee", "Owner"},
            types={"todo_assignee", "assignee", "user"},
        ),
    )

    if not field:
        return None

    users = field.get("user") or []

    if isinstance(users, dict):
        users = [users]

    if users:
        raw = users[0]

        if isinstance(raw, dict):
            return str(
                raw.get("id")
                or raw.get("user_id")
                or raw.get("value")
            )

        return str(raw)

    value = field.get("value")

    if isinstance(value, list):
        value = value[0] if value else None

    if isinstance(value, dict):
        value = (
            value.get("id")
            or value.get("user_id")
            or value.get("value")
        )

    return str(value).strip() if value else None


@lru_cache(maxsize=512)
def user_display_name(user_id):
    if not user_id:
        return None

    try:
        data = checked(
            client().users_info(user=user_id),
            "Read Slack user",
        )

        user = data.get("user") or {}
        profile = user.get("profile") or {}

        name = (
            profile.get("display_name")
            or profile.get("real_name")
            or user.get("real_name")
            or user.get("name")
        )

        return str(name).lstrip("@") if name else None

    except Exception:
        return None


def extract_assignee(item, schema):
    return user_display_name(
        extract_assignee_id(item, schema)
    )


def extract_due_date(item, schema):
    """Extract the due-date for an item as a YYYY-MM-DD string, or None.

    Handles all Slack API date representations robustly:
    - Plain date string: "2026-09-17"
    - ISO datetime string: "2026-09-17T00:00:00Z" or "2026-09-17T05:45:00+05:45"
    - Unix timestamp (int/float): seconds since epoch → converted via Asia/Kathmandu
    - Nested list (Slack lists API): ["2026-09-17"]
    - Nested dict: {"date": "2026-09-17"}
    """
    from datetime import timezone

    field = _item_field(
        item,
        column(
            schema,
            keys=DUE_KEYS,
            names={"Due Date", "Date"},
            types={"todo_due_date", "due_date", "date"},
        ),
    )

    if not field:
        return None

    # Slack's lists API nests the actual date under "date" key (as a list) or "value"
    value = field.get("date", field.get("value"))

    # Unwrap list → take first element
    if isinstance(value, list):
        value = value[0] if value else None

    # Unwrap dict → prefer "date", then "start", then "value"
    if isinstance(value, dict):
        value = value.get("date") or value.get("start") or value.get("value")

    if value is None:
        return None

    # Unix timestamp (int or float)
    if isinstance(value, (int, float)) and value > 0:
        from zoneinfo import ZoneInfo as _ZI
        try:
            return datetime.fromtimestamp(value, tz=_ZI("Asia/Kathmandu")).date().isoformat()
        except Exception:
            return None

    value = str(value).strip()
    if not value or value in ("-1", "null", "None"):
        return None

    # ISO datetime with time component — strip the time, keep the date
    # e.g. "2026-09-17T00:00:00Z" → "2026-09-17"
    if "T" in value:
        value = value.split("T")[0]

    # Validate the remaining string is a proper YYYY-MM-DD date
    try:
        date.fromisoformat(value)
        return value
    except ValueError:
        return None



def extract_status(item, schema):
    field = _item_field(
        item,
        column(
            schema,
            keys=STATUS_KEYS,
            names={"Status", "State"},
            types={"select", "multi_select"},
        ),
    )

    if not field:
        return (
            "completed"
            if extract_completed(item, schema)
            else "open"
        )

    selected = (
        field.get("select")
        or field.get("multi_select")
        or field.get("value")
    )

    if isinstance(selected, list):
        selected = selected[0] if selected else None

    choices = (
        (
            column(
                schema,
                keys=STATUS_KEYS,
                names={"Status", "State"},
            )
            or {}
        ).get("options")
        or {}
    ).get("choices") or []

    for choice in choices:
        if isinstance(choice, dict) and selected in {
            choice.get("id"),
            choice.get("value"),
        }:
            return (
                config.normalize_status(
                    choice.get("label") or choice.get("name")
                )
                or str(
                    choice.get("label")
                    or choice.get("value")
                )
            )

    return (
        config.normalize_status(selected)
        or str(selected or "open")
    )


def _norm(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip().casefold()


def _strip_punctuation(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s)


def find_matches(items, query, schema, assignee=None):
    """
    Resolve *query* against Slack List items using a tiered matching strategy:
      1. Exact normalised match  (highest confidence)
      2. Word-boundary prefix match (e.g. "Client Report" matches "Client Report Draft" only if
         every word in the query appears as a contiguous prefix of the title)
      3. Strict token subset – all query tokens present as whole words in the title
    Tier 1 wins outright; tiers 2 and 3 accumulate candidates.

    False-positive prevention: "docs check" must NOT match "onboarding docs" or "report".
    The query must be specific enough to constrain a single task.
    """
    if not query:
        return []

    q_norm = _norm(_strip_punctuation(query))
    q_tokens = set(q_norm.split())

    exact, tier2, tier3 = [], [], []

    for item in items:
        if assignee and extract_assignee_id(item, schema) != assignee:
            continue

        raw_name = extract_item_name(item, schema)
        n_norm = _norm(_strip_punctuation(raw_name))
        n_tokens = set(n_norm.split())

        # Tier 1 — exact
        if q_norm == n_norm:
            exact.append(item)
            continue

        # Tier 2 — query is a word-boundary prefix of the title
        if n_norm.startswith(q_norm + " ") or n_norm.startswith(q_norm):
            tier2.append(item)
            continue

        # Tier 3 — all query tokens are whole words in the title AND query
        # covers at least half the title tokens (prevents "docs" matching "docs check review")
        if q_tokens and q_tokens.issubset(n_tokens):
            coverage = len(q_tokens) / max(len(n_tokens), 1)
            if coverage >= 0.5:
                tier3.append(item)

    if exact:
        return exact
    if tier2:
        return tier2
    if tier3:
        return tier3
    return []



def find_user_id(user_text):
    if not user_text:
        return None

    value = str(user_text).strip()

    match = re.search(
        r"<@([UW][A-Z0-9]+)(?:\|[^>]+)?>",
        value,
        re.I,
    )

    if match:
        return match.group(1).upper()

    if re.fullmatch(r"[UW][A-Z0-9]+", value, re.I):
        return value.upper()

    value = value.lstrip("@").strip()
    cursor = None

    while True:
        kwargs = {"limit": 200}

        if cursor:
            kwargs["cursor"] = cursor

        data = checked(
            client().users_list(**kwargs),
            "Find Slack user",
        )

        for user in data.get("members", []) or []:
            if user.get("deleted") or user.get("is_bot"):
                continue

            profile = user.get("profile") or {}

            candidates = [
                user.get("name"),
                user.get("real_name"),
                profile.get("display_name"),
                profile.get("real_name"),
            ]

            if any(
                candidate and _norm(candidate) == _norm(value)
                for candidate in candidates
            ):
                return user.get("id")

        cursor = (
            data.get("response_metadata") or {}
        ).get("next_cursor") or ""

        if not cursor:
            return None


def find_priority_option(schema, priority):
    wanted = config.normalize_priority(priority)

    field = column(
        schema,
        keys=PRIORITY_KEYS,
        names={"Priority"},
        types={"select", "multi_select"},
    )

    if not field:
        return None

    for choice in (
        (field.get("options") or {}).get("choices") or []
    ):
        if not isinstance(choice, dict):
            continue

        if (
            config.normalize_priority(choice.get("label"))
            == wanted
            or config.normalize_priority(choice.get("value"))
            == wanted
        ):
            return choice.get("value") or choice.get("id")

    return None


def _name_cell(column_id, value):
    return {
        "column_id": column_id,
        "rich_text": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {
                                "type": "text",
                                "text": value,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _resolve_assignee(value):
    user_id = find_user_id(value)

    if not user_id:
        raise ValueError(
            f"I couldn't resolve the assignee {value!r}."
        )

    return user_id


def _write_cell(schema, field, value):
    if field == "name":
        col = column(
            schema,
            keys=NAME_KEYS,
            names={"Name", "Title", "Task", "Task Name"},
            types={"text", "rich_text"},
        )

        if not col:
            raise RuntimeError(
                "This List has no task-name field."
            )

        return _name_cell(
            column_id(col),
            str(value).strip(),
        )

    if field == "priority":
        col = column(
            schema,
            keys=PRIORITY_KEYS,
            names={"Priority"},
            types={"select", "multi_select"},
        )

        option = find_priority_option(schema, value)

        if not col or not option:
            raise RuntimeError(
                "This List has no usable Priority field/options."
            )

        return {
            "column_id": column_id(col),
            "select": [option],
        }

    if field == "assignee":
        col = column(
            schema,
            keys=ASSIGNEE_KEYS,
            names={"Assignee", "Owner"},
            types={"todo_assignee", "assignee", "user"},
        )

        if not col:
            raise RuntimeError(
                "This List has no Assignee field."
            )

        return {
            "column_id": column_id(col),
            "user": [_resolve_assignee(value)],
        }

    if field == "due_date":
        col = column(
            schema,
            keys=DUE_KEYS,
            names={"Due Date", "Date"},
            types={"todo_due_date", "due_date", "date"},
        )

        if not col:
            raise RuntimeError(
                "This List has no Due Date field."
            )

        date.fromisoformat(str(value))

        return {
            "column_id": column_id(col),
            "date": [str(value)],
        }

    if field == "completed":
        col = column(
            schema,
            keys=COMPLETED_KEYS,
            names={"Completed"},
            types={"todo_completed", "completed", "checkbox"},
        )

        if not col:
            raise RuntimeError(
                "This List has no Completed field."
            )

        return {
            "column_id": column_id(col),
            "checkbox": bool(value),
        }

    if field == "status":
        col = column(
            schema,
            keys=STATUS_KEYS,
            names={"Status", "State"},
            types={"select", "multi_select"},
        )

        if not col:
            raise RuntimeError(
                "This List has no Status field; use completed/open instead."
            )

        wanted = config.normalize_status(value)

        if not wanted:
            raise ValueError(
                "Status must be open, in progress, or completed."
            )

        choices = (
            (col.get("options") or {}).get("choices") or []
        )

        for choice in choices:
            if not isinstance(choice, dict):
                continue

            candidate = (
                choice.get("label")
                or choice.get("name")
                or choice.get("value")
            )

            if config.normalize_status(candidate) == wanted:
                return {
                    "column_id": column_id(col),
                    "select": [
                        choice.get("value") or choice.get("id")
                    ],
                }

        raise RuntimeError(
            f"Status option {wanted!r} is not available in this List."
        )

    raise ValueError(
        f"Unsupported field: {field}"
    )


def create_action_item(
    task_name,
    priority=None,
    assignee=None,
    due_date=None,
    context=None,
    list_id=None,
):
    if not config.has_permission(context, "create"):
        raise PermissionError(
            "You do not have permission to create action items."
        )

    actual = list_id or context.list_id
    schema = get_list_schema(actual)

    fields = [
        _write_cell(schema, "name", task_name)
    ]

    if priority:
        fields.append(
            _write_cell(schema, "priority", priority)
        )

    if assignee:
        fields.append(
            _write_cell(schema, "assignee", assignee)
        )

    if due_date:
        fields.append(
            _write_cell(schema, "due_date", due_date)
        )

    data = checked(
        client().api_call(
            api_method="slackLists.items.create",
            http_verb="POST",
            json={
                "list_id": actual,
                "initial_fields": fields,
            },
        ),
        "Create action item",
    )

    if not data.get("item"):
        raise RuntimeError(
            "Slack did not return the created action item."
        )

    return data["item"]


def update_action_item_field(
    item_id,
    field,
    value,
    context=None,
    list_id=None,
):
    actual = list_id or context.list_id

    permission = (
        "complete"
        if field == "completed"
        else config.FIELD_PERMISSION.get(field)
    )

    if (
        not permission
        or not config.has_permission(context, permission)
    ):
        raise PermissionError(
            f"Your role cannot edit the {field.replace('_', ' ')} field."
        )

    schema = get_list_schema(actual)
    cell = _write_cell(schema, field, value)
    cell["row_id"] = item_id

    return checked(
        client().api_call(
            api_method="slackLists.items.update",
            http_verb="POST",
            json={
                "list_id": actual,
                "cells": [cell],
            },
        ),
        "Update action item",
    )


def complete_action_item(
    item_id,
    context=None,
    list_id=None,
):
    return update_action_item_field(
        item_id,
        "completed",
        True,
        context,
        list_id,
    )


def reopen_action_item(
    item_id,
    context=None,
    list_id=None,
):
    return update_action_item_field(
        item_id,
        "completed",
        False,
        context,
        list_id,
    )


def delete_action_item(
    item_id,
    context=None,
    list_id=None,
):
    if not config.has_permission(context, "delete"):
        raise PermissionError(
            "You do not have permission to delete action items."
        )

    actual = list_id or context.list_id

    return checked(
        client().api_call(
            api_method="slackLists.items.delete",
            http_verb="POST",
            json={
                "list_id": actual,
                "id": item_id,
            },
        ),
        "Delete action item",
    )