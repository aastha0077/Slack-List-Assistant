"""Read-only team calendar, deadline intelligence, and global team clock."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import os
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class MemberClock:
    user_id: str
    name: str
    timezone_name: str | None
    location: str | None
    local_time: datetime | None
    utc_offset: str | None
    working_start: time | None
    working_end: time | None
    availability: str


def parse_request(text):
    """Recognize explicit calendar/clock language deterministically."""
    value = re.sub(r"\s+", " ", str(text or "")).strip().rstrip(".?!")
    lower = value.casefold()
    exact_modes = (
        (r"(?:show\s+)?(?:the\s+)?team\s+calendar", "team"),
        (r"(?:show\s+)?(?:my\s+|the\s+)?calendar", "week"),
        (r"(?:show\s+)?today(?:'s)?\s+(?:calendar|deadlines)", "today"),
        (r"(?:show\s+)?(?:the\s+)?(?:weekly|week)\s+calendar", "week"),
        (r"(?:show\s+)?(?:the\s+)?(?:monthly|month)\s+calendar", "month"),
        (r"(?:show\s+)?(?:the\s+)?calendar\s+for\s+upcoming\s+deadlines", "upcoming"),
        (r"(?:show\s+)?overdue\s+calendar", "overdue"),
        (r"what\s+does\s+the\s+upcoming\s+week\s+look\s+like", "week"),
        (r"who\s+has\s+deadlines\s+tomorrow", "tomorrow"),
        (r"(?:show\s+)?calendar\s+deadline\s+(?:conflicts|pressure)", "pressure"),
        (r"(?:show\s+)?(?:the\s+)?(?:team\s+)?time\s*zones?", "clock"),
        (r"(?:show\s+)?(?:the\s+)?(?:global\s+)?team\s+clock", "clock"),
        (r"who\s+is\s+(?:currently\s+)?(?:working|within\s+working\s+hours)(?:\s+right\s+now)?", "availability"),
        (r"who\s+is\s+outside\s+working\s+hours", "availability"),
        (r"when\s+can\s+i\s+meet\s+with\s+(?:the\s+)?team", "coordination"),
        (r"what\s+is\s+the\s+best\s+time\s+to\s+coordinate\s+with\s+(?:the\s+)?team", "coordination"),
    )
    for pattern, mode in exact_modes:
        if re.fullmatch(pattern, lower):
            return {"intent": "calendar", "calendar_mode": mode}
    person_time = re.fullmatch(r"what\s+time\s+is\s+it\s+for\s+(.+)", value, re.I)
    if person_time:
        return {"intent": "calendar", "calendar_mode": "clock",
                "calendar_member": person_time.group(1).strip()}
    return None


def _json_env(name):
    try:
        value = json.loads(os.getenv(name, "{}") or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _configured(mapping, user_id, name):
    for key in (user_id, name, str(name).casefold()):
        if key in mapping:
            return mapping[key]
    folded = {str(key).casefold(): value for key, value in mapping.items()}
    return folded.get(str(name).casefold())


def _clock_offset(value):
    offset = value.utcoffset()
    if offset is None:
        return None
    minutes = int(offset.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "−"
    minutes = abs(minutes)
    return f"UTC{sign}{minutes // 60}:{minutes % 60:02d}"


def _parse_working_hours(value):
    if isinstance(value, str) and "-" in value:
        start, end = value.split("-", 1)
    elif isinstance(value, dict):
        start, end = value.get("start"), value.get("end")
    else:
        return None, None
    try:
        return time.fromisoformat(str(start).strip()), time.fromisoformat(str(end).strip())
    except (TypeError, ValueError):
        return None, None


def _availability(local, start, end):
    if not local or not start or not end:
        return "Working hours not configured"
    current = local.time().replace(second=0, microsecond=0)
    inside = start <= current < end if start <= end else current >= start or current < end
    current_minutes = current.hour * 60 + current.minute
    start_minutes = start.hour * 60 + start.minute
    end_minutes = end.hour * 60 + end.minute
    near = min(abs(current_minutes - start_minutes), abs(current_minutes - end_minutes)) <= 60
    if near:
        return "Near working-hours boundary"
    return "Working hours" if inside else "Outside working hours"


def member_clocks(members, now=None):
    now = now or datetime.now(timezone.utc)
    zones = _json_env("TEAM_TIMEZONES_JSON")
    locations = _json_env("TEAM_LOCATIONS_JSON")
    hours = _json_env("TEAM_WORKING_HOURS_JSON")
    clocks = []
    for member in members:
        profile = member.get("profile") or {}
        user_id = str(member.get("id") or "")
        name = (profile.get("display_name") or member.get("real_name") or
                profile.get("real_name") or member.get("name") or "Team member")
        zone_name = (_configured(zones, user_id, name) or member.get("tz") or
                     profile.get("tz"))
        local = None
        if zone_name:
            try:
                local = now.astimezone(ZoneInfo(str(zone_name)))
            except ZoneInfoNotFoundError:
                zone_name = None
        start, end = _parse_working_hours(_configured(hours, user_id, name))
        clocks.append(MemberClock(
            user_id, name, str(zone_name) if zone_name else None,
            _configured(locations, user_id, name) or member.get("tz_label"),
            local, _clock_offset(local) if local else None, start, end,
            _availability(local, start, end)))
    return clocks


def filter_tasks(tasks, mode, today):
    pending = [task for task in tasks if not task.completed]
    if mode == "today":
        return [task for task in pending if task.due_date == today]
    if mode == "tomorrow":
        return [task for task in pending if task.due_date == today + timedelta(days=1)]
    if mode == "week":
        return [task for task in pending if task.due_date and today <= task.due_date <= today + timedelta(days=6)]
    if mode == "month":
        return [task for task in pending if task.due_date and
                (task.due_date.year, task.due_date.month) == (today.year, today.month)]
    if mode == "overdue":
        return [task for task in pending if task.due_date and task.due_date < today]
    if mode == "upcoming":
        return [task for task in pending if task.due_date and today <= task.due_date <= today + timedelta(days=30)]
    return [task for task in pending if task.due_date]


def render_calendar(tasks, all_tasks, mode, today, name_for_user, *,
                    show_owner=True, show_priority=True):
    labels = {"today": "TODAY", "tomorrow": "TOMORROW", "week": "THIS WEEK",
              "month": today.strftime("%B %Y").upper(), "upcoming": "UPCOMING 30 DAYS",
              "overdue": "OVERDUE", "team": "TEAM", "pressure": "DEADLINE PRESSURE"}
    selected = sorted(filter_tasks(tasks, mode, today), key=lambda task: (
        task.due_date or date.max, {"P1": 1, "P2": 2, "P3": 3, "P4": 4}.get(task.priority, 9),
        task.name.casefold()))
    pending = [task for task in all_tasks if not task.completed]
    overdue = [task for task in pending if task.due_date and task.due_date < today]
    due_today = [task for task in pending if task.due_date == today]
    due_week = [task for task in pending if task.due_date and today <= task.due_date <= today + timedelta(days=6)]
    p1 = [task for task in selected if show_priority and task.priority == "P1"]
    lines = [f"*TEAM CALENDAR — {labels.get(mode, 'TEAM')}*", "",
             f"*{len(selected)} deadlines* · {len(overdue)} overdue · {len(due_today)} due today · {len(due_week)} this week"]
    if not selected:
        lines.extend(("", "_No authorized deadlines match this calendar view._"))
        return "\n".join(lines)
    grouped = {}
    for task in selected:
        grouped.setdefault(task.due_date, []).append(task)
    for due, values in grouped.items():
        if due < today:
            day_label = f"OVERDUE · {due.strftime('%a %b %d').upper()}"
        elif due == today:
            day_label = f"TODAY · {due.strftime('%a %b %d').upper()}"
        elif due == today + timedelta(days=1):
            day_label = f"TOMORROW · {due.strftime('%a %b %d').upper()}"
        else:
            day_label = due.strftime("%A · %b %d").upper()
        lines.extend(("", f"*{day_label}*"))
        for task in values[:12]:
            priority_label = f"[{task.priority}]" if show_priority and task.priority else "[TASK]"
            owners = ((", ".join(name_for_user(owner) for owner in task.owner_ids) or "Unassigned")
                      if show_owner else "Owner restricted")
            status = ("Overdue" if due < today else "Due today" if due == today
                      else task.priority if show_priority else "Upcoming")
            lines.append(f"{priority_label} *{task.name}*\n   {owners} · {status}")
        if len(values) > 12:
            lines.append(f"_…and {len(values) - 12} more deadlines_ ")
    collisions = [(due, values) for due, values in grouped.items() if len(values) >= 3]
    pressure = {}
    if show_owner:
        for task in selected:
            for owner in task.owner_ids or ("unassigned",):
                weight = (3 if show_priority and task.priority == "P1"
                          else 2 if show_priority and task.priority == "P2" else 1)
                pressure[owner] = pressure.get(owner, 0) + weight
    lines.extend(("", "*DEADLINE INTELLIGENCE*",
                  (f"• {len(p1)} high-priority deadline{'s' if len(p1) != 1 else ''} in this view"
                   if show_priority else "• Priority details are restricted for this view"),
                  f"• {len(collisions)} date collision{'s' if len(collisions) != 1 else ''} with 3+ tasks"))
    if pressure:
        owner, score = max(pressure.items(), key=lambda row: row[1])
        label = "Unassigned" if owner == "unassigned" else name_for_user(owner)
        lines.append(f"• Highest deadline pressure: *{label}* · score {score}")
    lines.append("\n_Read-only calendar · No task changes were made._")
    return "\n".join(lines)


def render_clock(clocks, requester_clock=None, coordination=False):
    title = "*TEAM TIME ZONES*"
    lines = [title, "", f"*{len(clocks)} team members* · Time-zone data is never guessed"]
    configured = []
    for clock in clocks:
        lines.extend(("", f"*{clock.name}*"))
        if not clock.timezone_name or not clock.local_time:
            lines.append("   Time zone not configured")
            continue
        configured.append(clock)
        relative = ""
        if requester_clock and requester_clock.local_time and requester_clock.user_id != clock.user_id:
            delta_minutes = int((clock.local_time.utcoffset() - requester_clock.local_time.utcoffset()).total_seconds() // 60)
            if delta_minutes:
                sign = "+" if delta_minutes > 0 else "−"
                absolute = abs(delta_minutes)
                relative = f" · {sign}{absolute // 60}h {absolute % 60:02d}m vs you"
            else:
                relative = " · Same offset as you"
        lines.append(f"   {clock.local_time.strftime('%a · %b %d · %I:%M %p')} · {clock.utc_offset}{relative}")
        location = f" · {clock.location}" if clock.location else ""
        lines.append(f"   {clock.timezone_name}{location}")
        if clock.working_start and clock.working_end:
            lines.append(f"   {clock.working_start.strftime('%I:%M %p')}–{clock.working_end.strftime('%I:%M %p')} · {clock.availability}")
        else:
            lines.append("   Working hours not configured")
    if coordination:
        lines.extend(("", "*⏰ Coordination Window*"))
        if configured and all(clock.working_start and clock.working_end for clock in configured):
            now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            found = None
            for step in range(0, 7 * 48):
                candidate = now + timedelta(minutes=30 * step)
                if all(clock.working_start <= candidate.astimezone(
                        ZoneInfo(clock.timezone_name)).time() < clock.working_end for clock in configured):
                    found = candidate
                    break
            lines.append(found.strftime("• Earliest shared working window starts %a %b %d at %H:%M UTC")
                         if found else "• No shared configured working window was found in the next 7 days.")
        else:
            lines.append("• Configure every member's time zone and working hours to calculate an overlap.")
    lines.append("\n_Read-only team clock · No locations or time zones were inferred._")
    return "\n".join(lines)
