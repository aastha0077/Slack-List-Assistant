import json
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

VALID_ROLES = {"admin", "manager", "member", "viewer"}

SLACK_LIST_CHANNEL_ID = os.getenv("SLACK_LIST_CHANNEL_ID", "C0C2F3XGW8G").strip()
ACTION_ITEMS_LIST_ID = os.getenv("ACTION_ITEMS_LIST_ID", "F0C1GLU2JVB").strip()
DEFAULT_LIST_ID = os.getenv("SLACK_LIST_ID", "").strip() or ACTION_ITEMS_LIST_ID
DEFAULT_ROLE = os.getenv("SLACK_DEFAULT_ROLE", "viewer").strip().lower()


def _load_json_env(name: str) -> dict:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def normalize_role(role: Optional[str]) -> str:
    value = str(role or "").strip().lower()
    return value if value in VALID_ROLES else "viewer"


def normalize_priority(priority: Optional[str]) -> Optional[str]:
    if priority is None:
        return None
    value = str(priority).strip().upper().replace("PRIORITY", "").replace(" ", "")
    return value if value in {"P1", "P2", "P3"} else None


def normalize_status(status: Optional[str]) -> Optional[str]:
    if status is None:
        return None
    value = str(status).strip().lower().replace("-", " ")
    aliases = {
        "done": "completed", "complete": "completed", "finished": "completed",
        "pending": "open", "todo": "open", "to do": "open", "incomplete": "open",
        "reopened": "open", "re opened": "open",
    }
    value = aliases.get(value, value)
    return value if value in {"open", "completed", "in progress"} else None


USER_ROLES = _load_json_env("SLACK_USER_ROLES_JSON")
CHANNEL_LISTS = _load_json_env("SLACK_CHANNEL_LISTS_JSON")
if SLACK_LIST_CHANNEL_ID and SLACK_LIST_CHANNEL_ID not in CHANNEL_LISTS:
    CHANNEL_LISTS[SLACK_LIST_CHANNEL_ID] = ACTION_ITEMS_LIST_ID

for user_id, role in USER_ROLES.items():
    if not str(user_id).strip() or normalize_role(role) != str(role).strip().lower():
        raise ValueError(f"Invalid role mapping for {user_id!r}")
for channel_id, list_id in CHANNEL_LISTS.items():
    if not str(channel_id).strip() or not str(list_id).strip():
        raise ValueError("Channel/List mappings must contain non-empty strings")

DEFAULT_ROLE = normalize_role(DEFAULT_ROLE)

PERMISSIONS = {
    "admin": {
        "create", "view", "update", "complete", "delete",
        "edit_priority", "edit_assignee", "edit_due_date", "edit_name", "edit_status",
    },
    "manager": {
        "create", "view", "update", "complete", "delete",
        "edit_priority", "edit_assignee", "edit_due_date", "edit_name", "edit_status",
    },
    "member": {
        "create", "view", "update", "complete",
        "edit_due_date", "edit_name", "edit_status",
    },
    "viewer": {"view"},
}

FIELD_PERMISSION = {
    "name": "edit_name", "task": "edit_name", "title": "edit_name",
    "priority": "edit_priority",
    "assignee": "edit_assignee", "assigned": "edit_assignee", "owner": "edit_assignee",
    "due": "edit_due_date", "due_date": "edit_due_date", "due date": "edit_due_date", "deadline": "edit_due_date",
    "status": "edit_status", "completed": "edit_status", "complete": "edit_status", "done": "edit_status",
}


def get_user_role(user_id: Optional[str]) -> str:
    return normalize_role(USER_ROLES.get(user_id, DEFAULT_ROLE))


def get_list_id_for_channel(channel_id: Optional[str]) -> str:
    return str(CHANNEL_LISTS.get(channel_id, DEFAULT_LIST_ID) or "").strip()


def has_permission(context, permission: str) -> bool:
    if context is None:
        return False
    return permission in PERMISSIONS.get(normalize_role(getattr(context, "role", None)), set())


def can_edit_field(context, field_name: str) -> bool:
    permission = FIELD_PERMISSION.get(str(field_name or "").strip().lower())
    return bool(permission and has_permission(context, permission))


@dataclass
class RequestContext:
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    team_id: Optional[str] = None
    role: Optional[str] = None
    list_id: Optional[str] = None
    thread_ts: Optional[str] = None

    def __post_init__(self):
        self.role = get_user_role(self.user_id) if not self.role else normalize_role(self.role)
        self.list_id = self.list_id or get_list_id_for_channel(self.channel_id)


def build_context(user_id=None, channel_id=None, team_id=None, thread_ts=None):
    return RequestContext(user_id=user_id, channel_id=channel_id, team_id=team_id, thread_ts=thread_ts)
