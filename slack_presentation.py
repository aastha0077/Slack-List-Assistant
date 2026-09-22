"""Slack mrkdwn presentation primitives with no task or query logic."""
from dataclasses import dataclass
from datetime import date
from html import escape


@dataclass(frozen=True)
class TaskRow:
    name: str
    assignee: str | None = None
    due_date: str | None = None
    show_due: bool = True
    priority: str | None = None
    status: str | None = None


def text(value):
    """Escape Slack control characters while retaining renderer-owned mrkdwn."""
    return escape(str(value or ""), quote=False)


def compact_date(value, today=None):
    if not value:
        return "No deadline"
    try:
        parsed = date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return text(value)
    today = today or date.today()
    label = parsed.strftime("%b %d").replace(" 0", " ")
    return label if parsed.year == today.year else f"{label}, {parsed.year}"


def task_line(row: TaskRow, position=None, bullet=False):
    prefix = "•" if bullet else (f"{position}." if position is not None else "•")
    details = [value for value in (
        row.assignee, compact_date(row.due_date) if row.show_due else None,
        row.priority, row.status,
    ) if value]
    suffix = " · ".join(text(value) for value in details)
    return f"{prefix} *{text(row.name or 'Unnamed task')}*" + (f" — {suffix}" if suffix else "")


def task_collection(rows, title="Action Items", numbered=True):
    rows = list(rows)
    count = len(rows)
    header = f"*{text(title)}* · {count} task{'s' if count != 1 else ''}"
    if not rows:
        return header + "\n_No action items found._"
    lines = [task_line(row, position=index if numbered else None, bullet=not numbered)
             for index, row in enumerate(rows, 1)]
    return header + "\n" + "\n".join(lines)


def task_detail(row: TaskRow, bullet=True):
    """Render one task as a Slack-native compact field block."""
    prefix = "• " if bullet else ""
    lines = [f"{prefix}*{text(row.name or 'Unnamed task')}*"]
    fields = [
        ("Assignee", row.assignee),
        ("Due", compact_date(row.due_date) if row.show_due else None),
        ("Priority", row.priority),
        ("Status", row.status),
    ]
    lines.extend(f"  • {label}: {text(value)}" for label, value in fields if value is not None)
    return "\n".join(lines)


def task_field_list(row: TaskRow):
    """Render labeled fields for requested/existing comparisons."""
    fields = [
        ("Task", row.name or "Unnamed task"),
        ("Assignee", row.assignee),
        ("Due", compact_date(row.due_date) if row.show_due else None),
        ("Priority", row.priority),
        ("Status", row.status),
    ]
    return "\n".join(f"• {label}: {text(value)}" for label, value in fields if value is not None)


def created_collection(rows, list_name="Action Items"):
    rows = list(rows)
    count = len(rows)
    header = f"*Action Items Created* · {count} task{'s' if count != 1 else ''}"
    blocks = [task_detail(row) for row in rows]
    body = "\n\n".join(blocks)
    subject = "Action item" if count == 1 else "Action items"
    final = f"{subject} created successfully and verified in *{text(list_name)}*."
    return header + (("\n\n" + body) if body else "") + "\n\n" + final


def bar(value, maximum, width=8):
    filled = 0 if maximum <= 0 else round(width * value / maximum)
    return "█" * filled + "░" * (width - filled)


def distribution(title, values):
    if not values:
        return f"*{text(title)}* · No reliable data available."
    maximum = max(values.values()) or 1
    return f"*{text(title)}*\n" + "\n".join(
        f"• {text(label)} · {bar(count, maximum)} {count}" for label, count in values.items())


def series(title, value):
    values = (value or {}).get("values") or {}
    if not values:
        return f"*{text(title)}* · No reliable timestamped records are available."
    maximum = max(values.values()) or 1
    return f"*{text(title)}*\n" + "\n".join(
        f"• {text(label)} · {bar(count, maximum)} {count}" for label, count in values.items())
