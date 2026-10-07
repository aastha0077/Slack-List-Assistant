import json
import os
from dataclasses import dataclass
from typing import Optional
try:
    from dotenv import load_dotenv
except ImportError:  # Lambda reads configuration from its environment.
    def load_dotenv():
        return False

load_dotenv()

VALID_ROLES = {"admin", "manager", "member", "viewer"}

SLACK_LIST_CHANNEL_ID = os.getenv("SLACK_LIST_CHANNEL_ID", "").strip()
ACTION_ITEMS_LIST_ID = os.getenv("ACTION_ITEMS_LIST_ID", "").strip()
DEFAULT_LIST_ID = os.getenv("SLACK_LIST_ID", "").strip() or ACTION_ITEMS_LIST_ID
DEFAULT_ROLE = os.getenv("SLACK_DEFAULT_ROLE", "viewer").strip().lower()
try:
    CONFIRMATION_THRESHOLD = max(2, int(os.getenv("SLACK_CONFIRMATION_THRESHOLD", "5")))
except ValueError:
    CONFIRMATION_THRESHOLD = 5


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
    raw = str(priority).strip().lower().replace("priority", "").strip()
    mapping = {
        "urgent": "P1", "critical": "P1", "highest": "P1", "p1": "P1",
        "high": "P1",
        "medium": "P2", "med": "P2", "p2": "P2",
        "normal": "P3", "p3": "P3",
        "low": "P3", "lowest": "P3", "p4": "P4",
    }
    if raw in mapping:
        return mapping[raw]
    value = str(priority).strip().upper().replace("PRIORITY", "").replace(" ", "")
    return value if value in {"P1", "P2", "P3", "P4"} else None


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

DEFAULT_PERMISSIONS = {
    "admin": {
        "create", "view", "update", "assign", "reassign", "complete", "reopen", "delete", "bulk",
        "view_others", "create_for_others", "update_others",
        "edit_priority", "edit_assignee", "edit_due_date", "edit_name", "edit_status",
    },
    "manager": {
        "create", "view", "update", "assign", "reassign", "complete", "reopen", "bulk",
        "view_others", "create_for_others", "update_others",
        "edit_priority", "edit_assignee", "edit_due_date", "edit_name", "edit_status",
    },
    "member": {
        "create", "view", "update", "assign", "reassign", "complete", "reopen",
        "edit_assignee", "edit_due_date", "edit_name", "edit_status",
    },
    "viewer": {"view"},
}

_permission_config = _load_json_env("SLACK_ROLE_PERMISSIONS_JSON")
PERMISSIONS = {role: set(values) for role, values in DEFAULT_PERMISSIONS.items()}
for role_key, values in _permission_config.items():
    role_name = str(role_key).split(":")[-1].lower()
    if role_name not in VALID_ROLES or not isinstance(values, list) or any(not isinstance(x, str) for x in values):
        raise ValueError("SLACK_ROLE_PERMISSIONS_JSON must map roles or team:roles to permission-name lists")
    PERMISSIONS[str(role_key)] = set(values)

DEFAULT_FIELD_PERMISSION = {
    "name": "edit_name", "task": "edit_name", "title": "edit_name",
    "priority": "edit_priority",
    "assignee": "edit_assignee", "assigned": "edit_assignee", "owner": "edit_assignee",
    "due": "edit_due_date", "due_date": "edit_due_date", "due date": "edit_due_date", "deadline": "edit_due_date",
    "status": "edit_status", "completed": "edit_status", "complete": "edit_status", "done": "edit_status",
}

_field_config = _load_json_env("SLACK_FIELD_CONTROLS_JSON")
FIELD_CONTROLS = {}
for field_name, edit_permission in DEFAULT_FIELD_PERMISSION.items():
    FIELD_CONTROLS[field_name] = {"read": "view", "edit": edit_permission}
for field_name in ("reviewer_attachments", "reviewer attachments"):
    FIELD_CONTROLS[field_name] = {"read": "view", "edit": None}
for field_key, control in _field_config.items():
    if not isinstance(control, dict):
        raise ValueError("SLACK_FIELD_CONTROLS_JSON values must be objects")
    read_permission = control.get("read", "view")
    edit_permission = control.get("edit")
    if not isinstance(read_permission, str) or (edit_permission is not None and not isinstance(edit_permission, str)):
        raise ValueError("Field controls require string read/edit permission names")
    FIELD_CONTROLS[str(field_key).strip().lower()] = {"read": read_permission, "edit": edit_permission}

# Compatibility for existing callers; all decisions below use FIELD_CONTROLS.
FIELD_PERMISSION = {name: control["edit"] for name, control in FIELD_CONTROLS.items() if control.get("edit")}


def configured_user_role(user_id: Optional[str], team_id: Optional[str] = None) -> Optional[str]:
    scoped = USER_ROLES.get(f"{team_id}:{user_id}") if team_id else None
    direct = USER_ROLES.get(user_id)
    return normalize_role(scoped or direct) if scoped or direct else None


def get_user_role(user_id: Optional[str], team_id: Optional[str] = None, slack_user: Optional[dict] = None) -> str:
    configured = configured_user_role(user_id, team_id)
    return configured if configured else DEFAULT_ROLE


def get_list_id_for_channel(channel_id: Optional[str], team_id: Optional[str] = None) -> str:
    scoped = CHANNEL_LISTS.get(f"{team_id}:{channel_id}") if team_id else None
    return str(scoped or CHANNEL_LISTS.get(channel_id, DEFAULT_LIST_ID) or "").strip()


def has_permission(context, permission: str) -> bool:
    if context is None:
        return False
    role = normalize_role(getattr(context, "role", None))
    team_id = getattr(context, "team_id", None)
    allowed = PERMISSIONS.get(f"{team_id}:{role}", PERMISSIONS.get(role, set())) if team_id else PERMISSIONS.get(role, set())
    return permission in allowed


def field_control(context, field_name: str) -> dict:
    name = str(field_name or "").strip().lower()
    list_id = getattr(context, "list_id", None) if context else None
    return FIELD_CONTROLS.get(f"{list_id}:{name}", FIELD_CONTROLS.get(name, {}))


def can_read_field(context, field_name: str) -> bool:
    permission = field_control(context, field_name).get("read")
    return bool(permission and has_permission(context, permission))


def can_edit_field(context, field_name: str) -> bool:
    permission = field_control(context, field_name).get("edit")
    return bool(permission and has_permission(context, permission))


@dataclass
class RequestContext:
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    team_id: Optional[str] = None
    role: Optional[str] = None
    list_id: Optional[str] = None
    thread_ts: Optional[str] = None
    msg_ts: Optional[str] = None

    def __post_init__(self):
        self.role = get_user_role(self.user_id, self.team_id) if not self.role else normalize_role(self.role)
        self.list_id = self.list_id or get_list_id_for_channel(self.channel_id, self.team_id)


def build_context(user_id=None, channel_id=None, team_id=None, thread_ts=None, msg_ts=None):
    return RequestContext(user_id=user_id, channel_id=channel_id, team_id=team_id, thread_ts=thread_ts, msg_ts=msg_ts)


def is_lambda_runtime(environ=None):
    values = os.environ if environ is None else environ
    return bool(values.get("AWS_LAMBDA_FUNCTION_NAME") or values.get("AWS_EXECUTION_ENV"))


def state_db_path(environ=None):
    """Keep local state beside the process; place relative Lambda state in /tmp."""
    values = os.environ if environ is None else environ
    configured = (values.get("STATE_DB") or "slack_list_state.sqlite3").strip()
    if is_lambda_runtime(values) and not os.path.isabs(configured):
        return os.path.join("/tmp", configured)
    return configured


def load_local_environment():
    if is_lambda_runtime():
        return False
    load_dotenv()
    return True


@dataclass(frozen=True)
class RuntimeConfig:
    bot_token: str
    signing_secret: str
    app_token: str | None
    lambda_runtime: bool

    @classmethod
    def from_environment(cls):
        return cls(
            bot_token=os.getenv("SLACK_BOT_TOKEN", "").strip(),
            signing_secret=os.getenv("SLACK_SIGNING_SECRET", "").strip(),
            app_token=os.getenv("SLACK_APP_TOKEN", "").strip() or None,
            lambda_runtime=is_lambda_runtime(),
        )

    def validate(self, *, socket_mode=False):
        missing = []
        if not self.bot_token:
            missing.append("SLACK_BOT_TOKEN")
        if socket_mode and not self.app_token:
            missing.append("SLACK_APP_TOKEN")
        if not socket_mode and not self.signing_secret:
            missing.append("SLACK_SIGNING_SECRET")
        if missing:
            raise RuntimeError("Missing required environment variables: " + ", ".join(missing))
