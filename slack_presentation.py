"""Compact Slack-native presentation primitives with no task or query logic."""
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from html import escape


@dataclass(frozen=True)
class TaskRow:
    name: str
    assignee: str | None = None
    due_date: str | None = None
    show_due: bool = True
    priority: str | None = None
    status: str | None = None
    completed_date: str | None = None


class ResponseComplexity(str, Enum):
    SIMPLE = "simple"
    STRUCTURED = "structured"
    ANALYTICAL = "analytical"
    COMPLEX = "complex"


class PresentationStrategy(str, Enum):
    COMPACT_CONFIRMATION = "compact_confirmation"
    TASK_CARDS = "task_cards"
    TASK_TABLE = "task_table"
    SECTIONED_REPORT = "sectioned_report"
    EMPTY_STATE = "empty_state"
    ERROR_STATE = "error_state"


class ResponseType(str, Enum):
    TASK_LIST = "task_list"
    TASK_SEARCH = "task_search"
    COMPLETED_TASKS = "completed_tasks"
    FOCUS = "focus"
    ANALYTICS = "analytics"
    EXPLANATION = "explanation"
    SIMULATION = "simulation"
    CONFIRMATION = "confirmation"
    ERROR = "error"
    EMPTY = "empty"


RESPONSE_TYPE_STRATEGIES = {
    ResponseType.TASK_LIST: PresentationStrategy.TASK_TABLE,
    ResponseType.TASK_SEARCH: PresentationStrategy.TASK_TABLE,
    ResponseType.COMPLETED_TASKS: PresentationStrategy.TASK_TABLE,
    ResponseType.FOCUS: PresentationStrategy.TASK_TABLE,
    ResponseType.ANALYTICS: PresentationStrategy.TASK_TABLE,
    ResponseType.EXPLANATION: PresentationStrategy.SECTIONED_REPORT,
    ResponseType.SIMULATION: PresentationStrategy.SECTIONED_REPORT,
    ResponseType.CONFIRMATION: PresentationStrategy.COMPACT_CONFIRMATION,
    ResponseType.ERROR: PresentationStrategy.ERROR_STATE,
    ResponseType.EMPTY: PresentationStrategy.EMPTY_STATE,
}


def task_collection_strategy(count, *, grouped=False):
    """Choose a deterministic mobile-safe layout from response size."""
    if count <= 0:
        return ResponseComplexity.SIMPLE, PresentationStrategy.EMPTY_STATE
    return ResponseComplexity.STRUCTURED, PresentationStrategy.TASK_TABLE


def text(value):
    """Escape Slack control characters while retaining renderer-owned mrkdwn."""
    return escape(str("" if value is None else value), quote=False)


def notice(title, message, *, next_step=None):
    """Render a compact, Slack-native informational or failure notice."""
    lines = [f"*{text(title)}*", "", text(message)]
    if next_step:
        lines.extend(("", f"*Next step:* {text(next_step)}"))
    return "\n".join(lines)


def empty_state(title, message, *, context=None):
    """Render one precise empty state without implying that all data is empty."""
    lines = [f"*{text(title)}*", "", text(message)]
    if context:
        lines.extend(("", text(context)))
    return "\n".join(lines)


def permission_denied(message):
    """Render a user-facing authorization denial without implementation detail."""
    return notice(
        "Permission denied", message,
        next_step="Ask a workspace administrator if you need access to this action.")


def clarification(message):
    """Render an actionable request for missing or ambiguous information."""
    return notice("More information needed", message)


def failure(message, *, next_step=None):
    """Render a verified-safe failure message; technical details belong in logs."""
    return notice("Action not completed", message, next_step=next_step)


def join_sections(*sections):
    """Join independently rendered Slack sections with one blank line.

    Callers provide semantic sections rather than managing boundary newlines.
    Internal line breaks inside task cards are preserved verbatim.
    """
    values = [str(section or "").strip() for section in sections if str(section or "").strip()]
    return "\n\n".join(values)


def render_section(title, *blocks):
    """Render a Slack-native heading and paragraph/item blocks."""
    heading = f"*{text(title)}*" if title else ""
    return join_sections(heading, *blocks)


def render_item_list(items):
    """Keep multi-line task items visually separate on desktop and mobile."""
    return join_sections(*(str(item or "").strip() for item in items))


def render_task_card(name, metadata=(), *, prefix=None, detail=None):
    """Render one mobile-first task card with dominant title and quiet detail."""
    title = f"*{text(name or 'Unnamed task')}*"
    if prefix:
        title = f"{text(prefix)} {title}"
    lines = [title]
    indent = "   " if prefix else "  "
    values = [text(value) for value in metadata if value not in {None, ""}]
    if values:
        lines.append(indent + " · ".join(values))
    if detail:
        lines.append(f"{indent}↳ {text(detail)}")
    return "\n".join(lines)


def render_slack_table(columns, rows, *, title=None, summary=None, max_width=78):
    """Render comparable records as one bounded, Slack-safe monospace table.

    ``columns`` is an ordered collection of headings and ``rows`` contains
    sequences with matching positions. Values are escaped, flattened to one
    line and truncated only after the final mobile-width budget is known.
    """
    headings = [_table_cell(value) or "—" for value in columns]
    values = [[_table_cell(value) or "—" for value in row] for row in rows]
    if not headings:
        raise ValueError("A Slack table needs at least one column.")
    if any(len(row) != len(headings) for row in values):
        raise ValueError("Slack table rows must match the declared columns.")
    natural = [max(len(headings[index]), *(len(row[index]) for row in values))
               for index in range(len(headings))]
    minimum = [max(len(heading), 12 if index == 0 else len(heading))
               for index, heading in enumerate(headings)]
    widths = list(natural)
    spacing = 2 * max(0, len(widths) - 1)
    while sum(widths) + spacing > max_width:
        candidates = [index for index, width in enumerate(widths)
                      if width > minimum[index]]
        if not candidates:
            break
        widest = max(candidates, key=lambda index: (widths[index] - minimum[index], widths[index]))
        widths[widest] -= 1

    def truncate(value, width):
        return value if len(value) <= width else value[:max(1, width - 1)].rstrip() + "…"

    def render(row):
        return "  ".join(truncate(value, widths[index]).ljust(widths[index])
                          for index, value in enumerate(row)).rstrip()

    table = "\n".join(("```", render(headings), "─" * min(max_width, sum(widths) + spacing),
                        *(render(row) for row in values), "```"))
    sections = []
    if title:
        sections.append(f"*{text(title)}*")
    if summary:
        sections.append(f"*{text(summary)}*")
    sections.append(table)
    return join_sections(*sections)


def focus_reason(value):
    """Turn deterministic risk language into concise display language."""
    reason = str(value or "").strip()
    match = re.fullmatch(r"P1 \+ overdue by (\d+) days?", reason, re.I)
    if match:
        days = int(match.group(1))
        return f"P1 priority + {days} day{'s' if days != 1 else ''} overdue"
    match = re.fullmatch(r"overdue by (\d+) days?", reason, re.I)
    if match:
        days = int(match.group(1))
        return f"{days} day{'s' if days != 1 else ''} overdue"
    return reason[:1].upper() + reason[1:] if reason else ""


def validate_slack_response(value, *, resolve_user=None):
    """Return Slack-safe text and non-sensitive response-quality issue codes.

    This final presentation gate only repairs defects that are safe to correct
    without interpreting task intent or changing business data. Structural
    concerns that require feature-specific judgment are reported for logs and
    regression tests, but are not silently rewritten.
    """
    rendered = str(value or "").strip()
    issues = []
    if not rendered:
        return failure(
            "No response content was available.",
            next_step="Please try the request again."), ("empty_response",)

    # Never expose a Python traceback, object representation, or an entire raw
    # API/model JSON document to a normal Slack user.
    if ("Traceback (most recent call last):" in rendered
            or re.search(r"<[^>]+ object at 0x[0-9a-f]+>", rendered, re.I)):
        return failure(
            "I couldn't safely present the result of that request.",
            next_step="Please try again. Technical details were recorded in the application logs."), (
                "internal_debug_output",)
    try:
        decoded = json.loads(rendered)
    except (TypeError, ValueError, json.JSONDecodeError):
        decoded = None
    if isinstance(decoded, (dict, list)):
        return failure(
            "I couldn't safely present the structured result of that request.",
            next_step="Please try again."), ("raw_json",)

    if re.search(r"\\\*\\\*[^\n]+?\\\*\\\*", rendered):
        issues.append("escaped_markdown_bold")
        rendered = re.sub(r"\\\*\\\*([^\n]+?)\\\*\\\*", r"*\1*", rendered)
    if re.search(r"\*\*[^*\n]+\*\*", rendered):
        issues.append("markdown_bold")
        rendered = re.sub(r"\*\*([^*\n]+)\*\*", r"*\1*", rendered)
    if re.search(r"\[:red_circle:\]\(https?://[^)]+\)", rendered, re.I):
        issues.append("emoji_image_link")
        rendered = re.sub(
            r"\[:red_circle:\]\(https?://[^)]+\)", "🔴", rendered,
            flags=re.I)
    if ":robot_face:" in rendered or "🤖" in rendered:
        issues.append("decorative_robot")
        rendered = rendered.replace(":robot_face:", "").replace("🤖", "").strip()

    def replace_user(match):
        user_id = match.group(0)
        name = resolve_user(user_id) if resolve_user else None
        issues.append("raw_user_id")
        return str(name or "Workspace member")

    # Slack mentions are intentionally excluded because Slack renders those as
    # names. This targets only bare IDs accidentally placed in visible text.
    rendered = re.sub(r"(?<!<@)\bU[A-Z0-9]{8,}\b", replace_user, rendered)
    if re.search(r"\bF[A-Z0-9]{8,}\b", rendered):
        issues.append("raw_list_id")
        rendered = re.sub(r"\bF[A-Z0-9]{8,}\b", "Action Items", rendered)

    if rendered.count("```") % 2:
        issues.append("unclosed_code_fence")
        rendered += "\n```"

    headings = [line.strip() for line in rendered.splitlines()
                if re.fullmatch(r"\*[^*\n]+\*", line.strip())]
    if len(headings) != len(set(headings)):
        issues.append("duplicate_section")
    lower = rendered.casefold()
    if ("no action items found" in lower or "no tasks found" in lower) and re.search(
            r"\b[1-9]\d*\s+(?:pending|completed|authorized)\b", lower):
        issues.append("contradictory_empty_state")
    if len(rendered) > 12000:
        issues.append("excessive_length")
    return rendered, tuple(dict.fromkeys(issues))


def _date_value(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def compact_date(value, today=None):
    if not value:
        return "No due date"
    parsed = _date_value(value)
    if parsed is None:
        return text(value)
    today = today or date.today()
    label = parsed.strftime("%b %d").replace(" 0", " ")
    return f"{label}, {parsed.year}"


def compact_priority(value):
    """Keep configured priority values intact and shorten an unset priority."""
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.casefold() in {"", "none", "not set", "no priority", "no priority assigned", "—"}:
        return "—"
    return normalized


def due_label(row: TaskRow, today=None):
    """Present a due date as a compact date or an immediately useful time state."""
    if not row.show_due:
        return None
    if not row.due_date:
        return "No due date"
    parsed = _date_value(row.due_date)
    if parsed is None:
        return text(row.due_date)
    today = today or date.today()
    if str(row.status or "").casefold() != "completed":
        if parsed < today:
            return f"🔴 Overdue ({compact_date(row.due_date, today)})"
        if parsed == today:
            return "🟡 Due today"
        if parsed == today + timedelta(days=1):
            return "Due tomorrow"
    return compact_date(row.due_date, today)


def _status_is_redundant(status, due):
    return (str(status or "").casefold() == "pending"
            and due in {"🟡 Due today", "Due tomorrow"}) or (
                str(status or "").casefold() == "pending"
                and str(due or "").startswith("🔴 Overdue"))


def task_summary(row: TaskRow, *, show_assignee=True, show_due=True,
                 show_priority=True, show_status=True, today=None):
    """Render task content without a bullet, suitable for any response type."""
    due = due_label(row, today) if show_due else None
    details = []
    if show_priority and row.priority is not None:
        details.append(compact_priority(row.priority))
    if show_assignee and row.assignee:
        details.append(row.assignee)
    if due:
        details.append(due)
    if show_status and row.status and not _status_is_redundant(row.status, due):
        details.append(row.status)
    suffix = " · ".join(text(value) for value in details if value is not None)
    return f"*{text(row.name or 'Unnamed task')}*" + (f" · {suffix}" if suffix else "")


def task_line(row: TaskRow, position=None, bullet=False, *, show_assignee=True,
              show_due=True, show_priority=True, show_status=True, today=None):
    prefix = "•" if bullet or position is None else f"{position}."
    return f"{prefix} " + task_summary(
        row, show_assignee=show_assignee, show_due=show_due,
        show_priority=show_priority, show_status=show_status, today=today)


def _title_rules(title, rows, show_assignee, show_due, show_status):
    normalized = str(title or "").casefold()
    if show_assignee is None:
        show_assignee = not normalized.startswith(("your ", "my ", "focus today", "today's focus"))
    if show_due is None:
        show_due = not any(value in normalized for value in (
            "focus today", "today's focus", "due today", "due tomorrow"))
    if show_status is None:
        statuses = {str(row.status or "").casefold() for row in rows if row.status}
        status_is_heading = (
            statuses == {"pending"} and any(value in normalized for value in (
                "pending", "focus today", "today's focus", "due today", "due tomorrow", "overdue"))
            or statuses == {"completed"} and "completed" in normalized)
        owner_implied = normalized.startswith(("your ", "my ")) and statuses == {"pending"}
        show_status = not (status_is_heading or owner_implied)
    return show_assignee, show_due, show_status


def _summary(rows, title, today):
    if len(rows) < 4:
        return ""
    values = []
    priorities = Counter(compact_priority(row.priority) for row in rows)
    named_priorities = [priority for priority in ("P1", "P2", "P3") if priorities[priority]]
    if len(named_priorities) > 1:
        values.extend(f"{priorities[priority]} {priority}" for priority in named_priorities)
    normalized_title = str(title or "").casefold()
    if "overdue" not in normalized_title:
        overdue = sum(bool(_date_value(row.due_date) and _date_value(row.due_date) < today)
                      and str(row.status or "").casefold() != "completed" for row in rows)
        if overdue:
            values.append(f"{overdue} overdue")
    if "today" not in normalized_title:
        due_today = sum(_date_value(row.due_date) == today
                        and str(row.status or "").casefold() != "completed" for row in rows)
        if due_today:
            values.append(f"{due_today} due today")
    return " · ".join(values)


def _due_group(row, today):
    if str(row.status or "").casefold() == "completed":
        return "Completed"
    parsed = _date_value(row.due_date)
    if parsed is None:
        return "No Due Date"
    if parsed < today:
        return "Overdue"
    if parsed == today:
        return "Due Today"
    return "Upcoming"


def _table_cell(value):
    """Keep untrusted values on one line without breaking Slack's code fence."""
    return text(value).replace("\r", " ").replace("\n", " ").replace("```", "''' ").strip()


def _table_due_label(row, today):
    if not row.show_due:
        return None
    parsed = _date_value(row.due_date)
    if not parsed:
        return "—"
    label = compact_date(row.due_date, today)
    if str(row.status or "").casefold() != "completed":
        if parsed < today:
            return f"🔴 {label}"
        if parsed == today:
            return "🟡 Today"
    return label


def _task_table(rows, *, show_assignee, show_due, show_status, today):
    """Render a compact, reliably aligned Slack mrkdwn table."""
    def task_name(row):
        name = row.name or "Unnamed task"
        return f"{name} ✓" if str(row.status or "").casefold() == "completed" else name

    columns = [("Task", task_name)]
    if any(row.priority is not None for row in rows):
        columns.append(("Priority", lambda row: compact_priority(row.priority) or "—"))
    if show_assignee:
        columns.append(("Owner", lambda row: row.assignee or "Unassigned"))
    completed_only = bool(rows) and all(
        str(row.status or "").casefold() == "completed" for row in rows)
    if completed_only:
        columns = [("Task", lambda row: row.name or "Unnamed task"),
                   ("Completed", lambda row: compact_date(row.completed_date, today)
                    if row.completed_date else "✓")]
    else:
        if show_due:
            columns.append(("Due", lambda row: _table_due_label(row, today) or "—"))
        if show_status:
            columns.append(("Status", lambda row: row.status or "—"))
    values = [[getter(row) for _, getter in columns] for row in rows]
    return render_slack_table([name for name, _ in columns], values)


def _collection_heading(title):
    normalized = str(title or "").casefold()
    if "search" in normalized:
        return "🔎 Task Search"
    if "current list state" in normalized:
        return "✅ Completed (current List state)"
    if "completed" in normalized:
        return "✅ Completed Tasks"
    if normalized.startswith(("your ", "my ")):
        return "📋 My Action Items"
    return f"📋 {title or 'Action Items'}"


def _collection_summary(rows, title, today):
    count = len(rows)
    normalized = str(title or "").casefold()
    statuses = {str(row.status or "").casefold() for row in rows if row.status}
    if "search" in normalized:
        first = f"{count} match{'es' if count != 1 else ''}"
    elif statuses == {"completed"} or "completed" in normalized:
        first = f"{count} completed"
    elif statuses == {"pending"} or "pending" in normalized or normalized.startswith(("your ", "my ")):
        first = f"{count} pending"
    else:
        first = f"{count} task{'s' if count != 1 else ''}"
    overdue = sum(bool(_date_value(row.due_date) and _date_value(row.due_date) < today)
                  and str(row.status or "").casefold() != "completed" for row in rows)
    due_today = sum(_date_value(row.due_date) == today
                    and str(row.status or "").casefold() != "completed" for row in rows)
    values = [first]
    if overdue:
        values.append(f"{overdue} overdue")
    if due_today:
        values.append(f"{due_today} due today")
    return " · ".join(values)


def _task_cards(indexed_rows, *, show_assignee, show_due, show_status, today):
    """Render small collections as readable mobile-friendly task cards."""
    blocks = []
    for position, row in indexed_rows:
        completed = str(row.status or "").casefold() == "completed"
        name = f"{row.name or 'Unnamed task'} ✓" if completed else (row.name or "Unnamed task")
        details = []
        if row.priority is not None:
            details.append(compact_priority(row.priority) or "—")
        if show_assignee:
            details.append(row.assignee or "Unassigned")
        due = due_label(row, today) if show_due else None
        if due:
            details.append(due)
        if show_status and row.status and not completed and not _status_is_redundant(row.status, due):
            details.append(row.status)
        blocks.append(render_task_card(name, details, prefix=f"{position}."))
    return blocks


def task_collection(rows, title="Action Items", numbered=None, *, show_assignee=None,
                    show_due=None, show_status=None, summary=True, group_due=False,
                    group_status=False, empty_message=None, today=None, strategy=None):
    """Render an adaptive, mobile-readable task collection."""
    rows = list(rows)
    count = len(rows)
    heading = _collection_heading(title)
    header = f"*{text(heading)}*"
    if not rows:
        return header + "\n\n" + (empty_message or "No authorized action items found.")
    today = today or date.today()
    show_assignee, show_due, show_status = _title_rules(
        title, rows, show_assignee, show_due, show_status)
    summary_text = _collection_summary(rows, title, today) if summary else ""

    useful_due_grouping = group_due and count >= 3 and any(
        _due_group(row, today) in {"Overdue", "Due Today"} for row in rows)
    grouping_requested = useful_due_grouping or group_status
    if grouping_requested:
        labels = ("Overdue", "Due Today", "Upcoming", "No Due Date", "Pending", "Completed")
        rank = {label: index for index, label in enumerate(labels)}
        bucket_labels = [
            _due_group(row, today) if useful_due_grouping else (
                "Completed" if str(row.status or "").casefold() == "completed" else "Pending")
            for row in rows
        ]
        # Never reorder a stored task view just for presentation: displayed
        # order is also the meaning of follow-ups such as "the first task".
        grouping_requested = [rank[label] for label in bucket_labels] == sorted(
            rank[label] for label in bucket_labels)
    if not grouping_requested:
        selected_strategy = PresentationStrategy(strategy) if strategy else PresentationStrategy.TASK_TABLE
        if selected_strategy == PresentationStrategy.TASK_CARDS:
            body = render_item_list(_task_cards(
                list(enumerate(rows, 1)), show_assignee=show_assignee,
                show_due=show_due, show_status=show_status, today=today))
        else:
            body = _task_table(
                rows, show_assignee=show_assignee, show_due=show_due,
                show_status=(show_status and "search" in str(title).casefold()), today=today)
        return join_sections(header, f"*{summary_text}*" if summary_text else None, body)

    sections = [header, f"*{summary_text}*" if summary_text else None]
    groups = {label: [] for label in labels}
    for index, row in enumerate(rows, 1):
        label = _due_group(row, today) if useful_due_grouping else (
            "Completed" if str(row.status or "").casefold() == "completed" else "Pending")
        groups[label].append((index, row))
    for label, group in groups.items():
        if not group:
            continue
        hide_due = label in {"Overdue", "Due Today"}
        selected_strategy = PresentationStrategy(strategy) if strategy else PresentationStrategy.TASK_TABLE
        if selected_strategy == PresentationStrategy.TASK_CARDS:
            body = render_item_list(_task_cards(
                group, show_assignee=show_assignee,
                show_due=show_due and not hide_due, show_status=False, today=today))
        else:
            body = _task_table(
                [row for _, row in group], show_assignee=show_assignee,
                show_due=show_due and not hide_due, show_status=False, today=today)
        sections.append(render_section(label, body))
    return join_sections(*sections)


def task_detail(row: TaskRow, bullet=True):
    """Compatibility helper for a labeled single-task detail view."""
    prefix = "• " if bullet else ""
    lines = [f"{prefix}*{text(row.name or 'Unnamed task')}*"]
    fields = [
        ("Assignee", row.assignee),
        ("Due", compact_date(row.due_date) if row.show_due else None),
        ("Priority", compact_priority(row.priority)),
        ("Status", row.status),
    ]
    lines.extend(f"  • {label}: {text(value)}" for label, value in fields if value is not None)
    return "\n".join(lines)


def task_field_list(row: TaskRow):
    """Compatibility helper for labeled requested/existing comparisons."""
    fields = [
        ("Task", row.name or "Unnamed task"),
        ("Assignee", row.assignee),
        ("Due", compact_date(row.due_date) if row.show_due else None),
        ("Priority", compact_priority(row.priority)),
        ("Status", row.status),
    ]
    return "\n".join(f"• {label}: {text(value)}" for label, value in fields if value is not None)


def created_collection(rows, list_name="Action Items", today=None):
    rows = list(rows)
    count = len(rows)
    header = "*✓ Action Item Created*" if count == 1 else f"*✓ Action Items Created* · {count}"
    if count == 1:
        row = rows[0]
        details = [compact_priority(row.priority) or "—", row.assignee or "Unassigned"]
        if row.show_due:
            details.append(compact_date(row.due_date, today))
        details.append(row.status or "Pending")
        body = (f"*{text(row.name or 'Unnamed task')}*\n\n"
                + " · ".join(text(value) for value in details if value is not None))
    else:
        body = "\n".join(task_line(
            row, position=index if count > 5 else None, bullet=count <= 5,
            show_status=False, today=today) for index, row in enumerate(rows, 1))
    subject = "Task" if count == 1 else "Tasks"
    return (header + (("\n\n" + body) if body else "")
            + f"\n\n{subject} created successfully and verified in *{text(list_name)}*.")


def task_conflict(requested: TaskRow, existing: TaskRow, today=None):
    """Compact duplicate-field conflict that never mutates either row."""
    owner = existing.assignee or requested.assignee
    identity = f"*{text(requested.name or existing.name or 'Unnamed task')}*"
    if owner:
        identity += f" · {text(owner)}"

    def values(row):
        priority = compact_priority(row.priority)
        due = compact_date(row.due_date, today) if row.show_due else None
        return " · ".join(text(value) for value in (priority, due) if value is not None)

    return ("*↔ Existing task differs*\n\n" + identity
            + f"\nRequested: {values(requested)}"
            + f"\nExisting: {values(existing)}"
            + "\n\nNo changes made — the existing task has different fields.")


def assignee_clarification(row: TaskRow, spoken_name, today=None):
    summary = task_summary(
        row, show_assignee=False, show_priority=False, show_status=False, today=today)
    return (f"*⚠ Assignee unclear*\n\n{summary}\n\n"
            f"I couldn't confidently match \"{text(spoken_name)}\" to a Slack member.\n"
            "Please mention the person or provide their Slack name.")


def bar(value, maximum, width=8):
    filled = 0 if maximum <= 0 else round(width * value / maximum)
    return "█" * filled + "░" * (width - filled)


def distribution(title, values):
    if not values:
        return f"*{text(title)}* · No reliable data available."
    return render_slack_table(
        ("Metric", "Count"), tuple((label, count) for label, count in values.items()),
        title=title)


def series(title, value):
    values = (value or {}).get("values") or {}
    if not values:
        return f"*{text(title)}* · No reliable timestamped records are available."
    return render_slack_table(
        ("Period", "Count"), tuple((label, count) for label, count in values.items()),
        title=title)
