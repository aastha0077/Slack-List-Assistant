import os
import re
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:cloud").strip()


# ---------------------------------------------------------------------------
# Timezone-aware date helpers
# ---------------------------------------------------------------------------

def _today_date():
    return datetime.now(ZoneInfo("Asia/Kathmandu")).date()


def _today() -> str:
    """Return today's date in Asia/Kathmandu as YYYY-MM-DD."""
    return _today_date().isoformat()


def _tomorrow() -> str:
    return (_today_date() + timedelta(days=1)).isoformat()


def _yesterday() -> str:
    return (_today_date() - timedelta(days=1)).isoformat()


def _resolve_natural_date(text: str) -> Optional[str]:
    t = text.casefold()
    today = _today_date()
    if re.search(r"\btoday\b", t): return today.isoformat()
    if re.search(r"\btomorrow\b", t): return (today + timedelta(days=1)).isoformat()
    if re.search(r"\byesterday\b", t): return (today - timedelta(days=1)).isoformat()
    if re.search(r"\bthis\s+week\b", t):
        days_ahead = (4 - today.weekday()) % 7 # Friday of this week
        return (today + timedelta(days=days_ahead)).isoformat()

    # Weekdays calculation
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for name, wday in weekdays.items():
        if re.search(rf"\bnext\s+{name}\b", t):
            days_ahead = (wday - today.weekday()) % 7 + 7
            return (today + timedelta(days=days_ahead)).isoformat()
        if re.search(rf"\b(?:this\s+)?{name}\b", t):
            days_ahead = (wday - today.weekday()) % 7
            if days_ahead == 0 and "today" not in t:
                days_ahead = 7
            return (today + timedelta(days=days_ahead)).isoformat()

    # Explicit ISO YYYY-MM-DD (with audit against anomalous years)
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            if today.year - 1 <= d.year <= today.year + 2:
                return d.isoformat()
        except Exception:
            pass

    months = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
              r"|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
    m = re.search(rf"\b({months})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", t, re.I) or \
        re.search(rf"\b(\d{{1,2}})\s+({months})\s+(\d{{4}})\b", t, re.I)
    if m:
        try:
            raw = m.group(0)
            for fmt in ("%B %d, %Y", "%B %d %Y", "%d %B %Y"):
                try:
                    d = datetime.strptime(raw.title().replace(",", ""), fmt.replace(",", "")).date()
                    if today.year - 1 <= d.year <= today.year + 2:
                        return d.isoformat()
                except ValueError:
                    pass
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Standardized Output Schema
# ---------------------------------------------------------------------------

def _empty_result(raw_text: str = "") -> Dict[str, Any]:
    return {
        "intent": "out_of_scope",
        "tasks": [],
        "task_name": None,
        "task_reference": None,
        "selection": None,
        "selection_index": None,
        "priority": None,
        "status": None,
        "status_filter": None,
        "assignee": None,
        "assignee_self": False,
        "due_date": None,
        "completed": None,
        "all_tasks": False,
        "changes": [],
        "confidence": 0.0,
        "needs_clarification": False,
        "clarification_question": None,
        "raw_text": raw_text.strip(),
    }


# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

# IMPORTANT: Never call str.format() on this string — it contains literal JSON braces.
# Use _inject_dates() below which uses safe str.replace().
SYSTEM_PROMPT = """\
You are a strict JSON intent parser for a Slack List assistant that manages action items/tasks.
Convert the user's natural-language request into a single JSON object exactly matching the schema below.

ALLOWED INTENTS: create | list | update | complete | reopen | delete | out_of_scope | clarify

KEY RULES:
1. Return ONLY valid JSON — no markdown fences, no extra text.
2. intent MUST be one of the allowed values.
3. task_name = the human-readable task title ONLY.
   - MUST NOT include @mention tokens, Slack user IDs, or meta-words like "task","assign","for","with".
   - When the command is "@User TASK_NAME [action]", task_name is TASK_NAME only.
   - Trailing word "task" / "item" that is NOT part of the real title must be stripped.
4. out_of_scope only if completely unrelated to tasks (weather, jokes, etc.)
5. Priorities: normalise to P1/P2/P3/P4. Map "urgent"/"high"->P1, "medium"->P2/P3, "low"->P4.
6. due_date: always YYYY-MM-DD.
   - "today" -> {today}
   - "tomorrow" -> {tomorrow}
   - Any other phrase -> compute YYYY-MM-DD.
7. For "update": output "changes" as [{"field": "...", "value": "..."}].
   Supported fields: assignee, priority, due_date, status, name.
   "to P2" / "priority to P3" -> changes=[{"field":"priority","value":"P2"}]
8. "my tasks" / first-person "I"/"me"/"my" in a LIST -> assignee_self=true.
   For delete/update/create, "my" means the task is assigned to the requester.
9. List filters: due_today (bool), overdue (bool), due_this_week (bool), completed (bool), query (str).
10. Pronouns ("this", "that", "the first one") -> task_name: "__LAST__", selection: "first"/"both"/"all".
11. If genuinely ambiguous -> intent: "clarify", question in "clarification".
12. All task-related queries -> intent "list" with appropriate filters.
13. assignee: exact text the user typed (display name or @mention). Do NOT invent user IDs.
14. "nearest deadline" / "next deadline" -> sort_by="due_date", sort_order="asc", limit=1.
15. "one task" / "single task" -> limit=1, sort_by="due_date", sort_order="asc".
16. "what should I work on next?" -> assignee_self=true, sort_by="due_date", sort_order="asc", limit=1, completed=false.
17. MULTIPLE TASKS: If the user provides multiple tasks in one message (bullet list, numbered list, or "task A and task B"):
    - Set intent="create".
    - Use "tasks" list (NOT task_name) with one entry per task.
    - Each entry: {"task_name": "...", "assignee": "...", "assignee_self": true/false, "priority": "...", "due_date": "YYYY-MM-DD"}.
    - Shared metadata (assignee, due_date, priority) applies to ALL entries unless overridden per task.
    - First-person ("I", "me", "my", "I want to work on") -> assignee_self=true for each task entry.
    - Omit null/false/missing fields inside each task entry.

JSON SCHEMA (omit null/false/missing fields):
{
  "intent": "create",
  "task_name": "...",
  "tasks": [{"task_name": "...", "assignee": "Praveen", "assignee_self": true, "priority": "P1", "due_date": "YYYY-MM-DD"}],
  "priority": "P1",
  "status": "open",
  "assignee": "Praveen",
  "assignee_self": true,
  "due_date": "YYYY-MM-DD",
  "completed": false,
  "query": "search term",
  "due_today": false,
  "overdue": false,
  "due_this_week": false,
  "selection": "first",
  "selection_count": 4,
  "limit": 1,
  "sort_by": "due_date",
  "sort_order": "asc",
  "changes": [{"field": "priority", "value": "P2"}],
  "clarification": "..."
}"""


def _inject_dates(prompt: str) -> str:
    """Safe date injection — str.replace leaves JSON braces untouched."""
    return prompt.replace("{today}", _today()).replace("{tomorrow}", _tomorrow())




# ---------------------------------------------------------------------------
# Deterministic local mutation parser
# ---------------------------------------------------------------------------
# Handles delete / update / complete / reopen without calling Ollama.
# Returns {} to fall through to Ollama when it cannot determine intent.

_DELETE_ANCHOR   = re.compile(r"^(?:delete|remove|cancel|drop|erase|trash)\b", re.I)
_UPDATE_ANCHOR   = re.compile(r"^(?:update|change|set|modify|edit|alter|fix|adjust|reschedule|reassign|rename)\b", re.I)
_REOPEN_ANCHOR   = re.compile(r"^(?:reopen|re-open|uncheck|uncomplete|undo\s+complete)\b", re.I)

# Comprehensive complete phrasing:
# 1. Imperative: complete X, finish X, mark X done/completed, close X
# 2. First-person: I finished X, I completed X, I did X, I have done X, I'm done with X, I wrapped up X, I've taken care of X, I already finished that
# 3. Done with X, Finished with X
_COMPLETE_PREFIX = re.compile(
    r"^(?:"
    r"i(?:'ve|'m|\s+have|\s+had|\s+am|\s+did|\s+was)?\s+(?:already\s+)?(?:done\s+with|finished\s+with|completed\s+with|finished|completed|done|wrapped\s+up|taken\s+care\s+of|took\s+care\s+of|closed)|"
    r"i\s+(?:did|finished|completed|closed)|"
    r"i(?:'ve|'m|\s+have|\s+had|\s+am)?\s+(?:done\s+with|finished\s+with|completed\s+with)|"
    r"done\s+with|finished\s+with|completed\s+with|"
    r"mark\s+(?:the\s+)?(?:\w+\s+)?(?:as\s+)?(?:done|complete|completed|finished|closed)|"
    r"mark|"
    r"complete|finish|done|close|closed"
    r")\b",
    re.I,
)

# Passive/state complete phrasing: "login bug is finished", "the login bug is complete", "login bug is done"
_COMPLETE_PASSIVE = re.compile(
    r"^(.+?)\s+(?:(?:is|are|was|were)\s+)?(?:done|finished|completed|complete|closed)\s*[.!?]*$",
    re.I,
)

# Slack mention <@UXXXXX> or <@UXXXXX|name>
_MENTION_RE = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]+)?>", re.I)
# Bare @Name (not inside <@...>)
_BARE_AT_RE  = re.compile(r"(?<![<|])@([A-Za-z]\w+)")

# Priority: p1-p4
_PRIORITY_RE = re.compile(r"\b(p[1-4])\b", re.I)
# Trailing " [priority] to P2" suffix on a task description
_PRIORITY_SUFFIX = re.compile(r"\s+(?:priority\s+)?to\s+(p[1-4])\s*$", re.I)
# "rename X to Y"
_RENAME = re.compile(r"^rename\s+(.+?)\s+to\s+(.+)$", re.I)
# Quoted task name
_QUOTED = re.compile(r'["“”‘’]([^"“”‘’]+)["“”‘’]')
# Trailing noise words
_TASK_NOISE = re.compile(r"\b(task|item|action\s+item|todo|to-do)s?\s*$", re.I)
# Pronoun references
_PRONOUN_TASKS = {
    "that", "that task", "that item", "that one",
    "this", "this task", "this item", "this one",
    "it", "the last one", "last one", "the last", "last",
    "the first one", "first one", "the first", "first",
    "the previous one", "previous one", "previous",
}

# Field-value pattern: "set/change FIELD [of TASK] to VALUE"
_FIELD_VALUE = re.compile(
    r"\b(?:set\s+)?(?:the\s+)?"
    r"(priority|due\s*date|due|deadline|assignee|assigned\s+to|owner|status|name|title)"
    r"\s+(?:of\s+.+?\s+)?(?:to|=|:)\s+(.+)$",
    re.I,
)

_FIELD_MAP = {
    "priority": "priority",
    "due date": "due_date", "due_date": "due_date", "due": "due_date",
    "deadline": "due_date",
    "assignee": "assignee", "assigned to": "assignee", "owner": "assignee",
    "status": "status", "name": "name", "title": "name",
}


def _extract_mention_from(text: str):
    """Return (token, uid_or_None) for the first Slack mention in text."""
    m = _MENTION_RE.search(text)
    if m:
        return m.group(0), m.group(1).upper()
    b = _BARE_AT_RE.search(text)
    if b:
        return "@" + b.group(1), None
    return None, None


def _strip_mention(text: str, token: str) -> str:
    if token:
        text = text.replace(token, " ")
    return re.sub(r"\s{2,}", " ", text).strip()


def _clean_task(name: str) -> str:
    name = name.strip(" .,;:\"'?!")
    # Strip leading "with "
    name = re.sub(r"^with\s+", "", name, flags=re.I).strip()
    # Strip leading articles
    name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.I).strip()
    # Strip trailing completion noise words
    name = re.sub(r"\s+(?:as\s+)?(?:done|complete|completed|finished|closed)\s*$", "", name, flags=re.I).strip()
    # Strip trailing noise words
    name = _TASK_NOISE.sub("", name).strip()
    # Strip leading articles again
    name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.I).strip()
    name = name.strip(" .,;:\"'?!")
    if name.lower() in _PRONOUN_TASKS:
        return "__LAST__"
    return name


_POSITIONAL_MAP = {
    "first": (1, "first"),
    "1st": (1, "first"),
    "first one": (1, "first"),
    "the first": (1, "first"),
    "the first one": (1, "first"),
    "second": (2, "second"),
    "2nd": (2, "second"),
    "second one": (2, "second"),
    "the second": (2, "second"),
    "the second one": (2, "second"),
    "third": (3, "third"),
    "3rd": (3, "third"),
    "third one": (3, "third"),
    "the third": (3, "third"),
    "the third one": (3, "third"),
    "fourth": (4, "fourth"),
    "4th": (4, "fourth"),
    "fourth one": (4, "fourth"),
    "the fourth": (4, "fourth"),
    "the fourth one": (4, "fourth"),
    "last": (-1, "last"),
    "last one": (-1, "last"),
    "the last": (-1, "last"),
    "the last one": (-1, "last"),
    "final": (-1, "last"),
    "final one": (-1, "last"),
    "the final one": (-1, "last"),
    "previous": (-1, "last"),
    "previous one": (-1, "last"),
    "the previous one": (-1, "last"),
    "both": (None, "both"),
    "both of them": (None, "both"),
    "all": (None, "all"),
    "all of them": (None, "all"),
    "everything": (None, "all"),
    "that": (None, "__LAST__"),
    "that task": (None, "__LAST__"),
    "that one": (None, "__LAST__"),
    "this": (None, "__LAST__"),
    "this task": (None, "__LAST__"),
    "this one": (None, "__LAST__"),
    "it": (None, "__LAST__"),
}

_POSITIONAL_RE = re.compile(
    r"\b(?:the\s+)?(first\s+one|second\s+one|third\s+one|fourth\s+one|last\s+one|final\s+one|previous\s+one|1st|2nd|3rd|4th|first|second|third|fourth|last|final|previous|both(?:\s+of\s+them)?|all(?:\s+of\s+them)?|everything|that\s+task|that\s+one|that|this\s+task|this\s+one|this|it)\b",
    re.I,
)


def _build_mutation_result(intent, task_name, text, body, t, changes=None, assignee_raw=None, assignee_self=False):
    r = _empty_result(text)
    r["intent"] = intent

    # Check positional reference in body, text, or task_name
    cand_str = (body or t or task_name or "").strip()
    pm = _POSITIONAL_RE.search(cand_str)
    if pm and (task_name in _PRONOUN_TASKS or task_name == "__LAST__" or not task_name
               or pm.group(1).lower() in _POSITIONAL_MAP):
        pos_key = pm.group(1).lower().strip()
        idx, ref = _POSITIONAL_MAP.get(pos_key, (None, pos_key))
        r["task_reference"] = ref
        r["selection"] = ref
        r["selection_index"] = idx
        r["task_name"] = "__LAST__"
    elif task_name:
        r["task_name"] = task_name

    if changes:
        r["changes"] = changes
    if assignee_raw:
        r["assignee"] = assignee_raw
    if assignee_self:
        r["assignee_self"] = True
    return r


def _local_parse_mutation(text: str) -> Dict[str, Any]:
    """
    Deterministic parser for mutation intents (delete/update/complete/reopen).
    Returns populated dict or {} to fall through to Ollama.
    """
    t = text.strip()

    is_complete = False
    passive_task = None

    if _COMPLETE_PREFIX.match(t):
        verb = "complete"
        is_complete = True
    elif _COMPLETE_PASSIVE.match(t) and not _READ_ANCHORS.match(t):
        m = _COMPLETE_PASSIVE.match(t)
        cand = _clean_task(m.group(1))
        # Ensure candidate is not a read query word
        if cand and cand.lower() not in ("what", "who", "all", "everything"):
            verb = "complete"
            is_complete = True
            passive_task = cand
    elif _DELETE_ANCHOR.match(t):
        verb = "delete"
    elif _UPDATE_ANCHOR.match(t):
        verb = "update"
    elif _REOPEN_ANCHOR.match(t):
        verb = "reopen"
    else:
        return {}

    # ── Multi-task complete handling ("I finished X and completed Y") ────────
    if is_complete and " and " in t.lower() and not passive_task:
        stripped_full = _COMPLETE_PREFIX.sub("", t, count=1).strip()
        parts = re.split(
            r"\s+and\s+(?:i\s+(?:also\s+)?(?:finished|completed|did|done)|completed|finished|done\s+with|mark\s+)?",
            stripped_full,
            flags=re.I,
        )
        cleaned_parts = [_clean_task(p) for p in parts if p.strip()]
        if len(cleaned_parts) >= 2 and all(len(p) >= 2 for p in cleaned_parts):
            return {
                "intent": "complete",
                "tasks": [{"task_name": p} for p in cleaned_parts],
                "raw_text": text,
            }

    # Strip the leading verb word(s)
    if is_complete:
        body = _COMPLETE_PREFIX.sub("", t, count=1).strip()
        body = re.sub(r"^(?:as\s+)?(?:done|complete|completed|finished)\s*", "", body, flags=re.I).strip()
    else:
        body = re.sub(r"^\S+\s*", "", t, count=1).strip()

    # Extract @mention / self-reference
    mention_token, _uid = _extract_mention_from(body)
    assignee_raw  = mention_token
    assignee_self = False

    if mention_token:
        body = _strip_mention(body, mention_token)
    elif re.match(r"^my\b", body, re.I):
        assignee_self = True
        body = re.sub(r"^my\s+(?:task|item|todo|action\s+item)?\s*", "", body, flags=re.I).strip()
    elif re.match(r"^(?:the\s+)?(?:task|item|action\s+item)\s+", body, re.I):
        body = re.sub(r"^(?:the\s+)?(?:task|item|action\s+item)\s+", "", body, flags=re.I).strip()

    def _build(intent, task_name, changes=None):
        return _build_mutation_result(intent, task_name, text, body, t, changes=changes, assignee_raw=assignee_raw, assignee_self=assignee_self)

    # ── complete / reopen — just need the task name ──
    if verb in ("complete", "reopen"):
        if is_complete and passive_task:
            return _build("complete", passive_task)
        q = _QUOTED.search(body)
        task_name = _clean_task(q.group(1) if q else body)
        if not task_name:
            return {}
        return _build(verb, task_name)

    # ── delete ──
    if verb == "delete":
        q = _QUOTED.search(body)
        task_name = _clean_task(q.group(1) if q else body)
        if not task_name:
            return {}
        return _build("delete", task_name)

    # ── update / change / set ──
    # Rename: "rename X to Y"
    rm = _RENAME.match(t)
    if rm:
        return _build("update", _clean_task(rm.group(1)), [{"field": "name", "value": rm.group(2).strip()}])

    # "set FIELD of TASK to VALUE" / "change FIELD to VALUE"
    fv = _FIELD_VALUE.match(t)
    if fv:
        raw_field = fv.group(1).strip().lower().replace(" ", "_")
        field = _FIELD_MAP.get(raw_field) or _FIELD_MAP.get(raw_field.replace("_", " "))
        raw_value = fv.group(2).strip()
        if field == "priority":
            pm = _PRIORITY_RE.match(raw_value)
            value = pm.group(1).upper() if pm else raw_value
        elif field == "due_date":
            value = _resolve_natural_date(raw_value) or raw_value
        else:
            value = raw_value
        if not field:
            return {}
        of_m = re.search(r"\bof\s+(.+?)\s+(?:to|=)", t, re.I)
        task_name = _clean_task(of_m.group(1)) if of_m else ""
        return _build("update", task_name, [{"field": field, "value": value}])

    # General: body is "TASK [FIELD] to VALUE"
    changes: list = []
    task_body = body

    # Strip due-date suffix
    due_m = re.search(r"\s+(?:due\s*(?:date)?|deadline)\s+to\s+(.+)$", task_body, re.I)
    if due_m:
        due_val = _resolve_natural_date(due_m.group(1)) or due_m.group(1).strip()
        changes.append({"field": "due_date", "value": due_val})
        task_body = task_body[:due_m.start()].strip()

    # Strip priority suffix ("priority to P2" or just "to P2")
    pri_m = _PRIORITY_SUFFIX.search(task_body)
    if pri_m:
        changes.append({"field": "priority", "value": pri_m.group(1).upper()})
        task_body = task_body[:pri_m.start()].strip()

    # Strip trailing field keyword
    task_body = re.sub(
        r"\s+(?:priority|due\s*date|deadline|assignee|status|name|title)\s*$",
        "", task_body, flags=re.I,
    ).strip()

    q = _QUOTED.search(task_body)
    task_name = _clean_task(q.group(1) if q else task_body)

    if not task_name and not changes:
        return {}

    return _build("update", task_name, changes or None)


# ---------------------------------------------------------------------------
# Deterministic local CREATE parser — multi-task support
# ---------------------------------------------------------------------------

# CREATE intent anchor phrases (must appear in the message OR message has bullet/numbered lines)
_CREATE_ANCHOR = re.compile(
    r"\b("
    r"add|create|i\s+have\s+(?:action\s+items?|tasks?|todo)|i\s+want\s+to\s+(?:work|add|create)|"
    r"i\s+need\s+to\s+(?:work|add|do|finish|complete|handle|tackle)|"
    r"my\s+(?:action\s+items?|tasks?)\s+(?:today|for|are|this)|"
    r"work\s+on|tasks?\s+(?:today|for\s+today|are)|"
    r"please\s+(?:add|create)|(?:can\s+you\s+)?(?:add|create)\s+(?:a\s+)?(?:task|item)|"
    r"i\s+am\s+(?:going\s+to|planning\s+to|working\s+on)|"
    r"i\s+will\s+(?:work\s+on|do|finish|complete|handle)"
    r")\b",
    re.I,
)

# Bullet list line: starts with *, -, •, or similar
_BULLET_LINE = re.compile(r"^\s*[\*\-\u2022\u25cf\u25e6>]\s+(.+)$")
# Numbered list line: starts with N. or N)
_NUMBERED_LINE = re.compile(r"^\s*\d+[.)\]]\s+(.+)$")

# Metadata extractors for shared attributes across all tasks in a CREATE message
_CREATE_ASSIGNEE = re.compile(
    r"\b(?:for|to|assign\s+to|assigned\s+to)\s+@?([A-Za-z][\w.]+)\b",
    re.I,
)
_CREATE_MENTION = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]+)?>")
_CREATE_PRIORITY = re.compile(r"\b(p[1-4])\b", re.I)
_CREATE_DUE_TODAY = re.compile(r"\btoday\b", re.I)
_CREATE_DUE_TOMORROW = re.compile(r"\btomorrow\b", re.I)
_CREATE_DUE_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# Sentence boundary splitter (for "Do X. Review Y. Send Z.")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z]|I\s)")

# Phrases to strip from the leading CREATE verb phrase so we get just the task body
_CREATE_VERB_STRIP = re.compile(
    r"^(?:"
    r"i\s+have\s+(?:a\s+few\s+|some\s+)?(?:action\s+items?|tasks?|todo)(?:\s+today)?(?:[.,:]?\s*i\s+(?:want|need)\s+to\s+(?:work\s+on|do))?|\s*"
    r"i\s+(?:want|need)\s+to\s+(?:work\s+on|do|finish|complete|handle|tackle)|\s*"
    r"i\s+(?:am\s+)?(?:going\s+to|planning\s+to|working\s+on)|\s*"
    r"i\s+will\s+(?:work\s+on|do|finish|complete|handle)|\s*"
    r"my\s+(?:action\s+items?|tasks?)\s+(?:today|for\s+today)?\s*(?:are)?|\s*"
    r"(?:please\s+)?(?:add|create)\s+(?:(?:a|an)\s+)?(?:(?:action\s+)?(?:item|task)\s+)?(?:called\s+|named\s+|of\s+)?|\s*"
    r"work\s+on|\s*"
    r"tasks?\s+(?:today|for\s+today)?\s*(?:are)?|\s*"
    r"action\s+items?\s*(?:are)?"
    r")\s*(?:[.,:]\s*)?",
    re.I,
)

# Trailing metadata noise to strip from individual task names
_TASK_META_SUFFIX = re.compile(
    r"\s*(?:,?\s*(?:both|all)\s+)?(?:p[1-4]|priority\s+p?[1-4]|today|tomorrow|by\s+\S+)\s*$",
    re.I,
)


def _extract_create_metadata(text: str):
    """
    Extract shared metadata (assignee, due_date, priority) from the full CREATE text.
    Returns a dict with only the fields that were found.
    """
    meta = {}

    # Priority
    pm = _CREATE_PRIORITY.search(text)
    if pm:
        meta["priority"] = pm.group(1).upper()
    else:
        np = re.search(r"\b(?:priority\s+)?(urgent|critical|highest|high|medium|normal|low|lowest)\b", text, re.I)
        if np:
            from config import normalize_priority
            p_val = normalize_priority(np.group(1))
            if p_val:
                meta["priority"] = p_val

    # Due date — explicit phrase first, then ISO / natural
    due_m = re.search(r"\b(?:due(?:\s+date)?|deadline|by)\s+([A-Za-z0-9\-\s,]+?)(?=\s+(?:for|to|assign(?:ed)?\s+to|priority|p[1-4])|$)", text, re.I)
    if due_m:
        nat_date = _resolve_natural_date(due_m.group(1))
        if nat_date:
            meta["due_date"] = nat_date

    if "due_date" not in meta:
        dm = _CREATE_DUE_DATE.search(text)
        if dm:
            meta["due_date"] = dm.group(1)
        elif _CREATE_DUE_TODAY.search(text):
            meta["due_date"] = _today()
        elif _CREATE_DUE_TOMORROW.search(text):
            meta["due_date"] = _tomorrow()
        else:
            nat = _resolve_natural_date(text)
            if nat:
                meta["due_date"] = nat

    # Self-assignment phrases: "to me", "for me", "assigned to me"
    if re.search(r"\b(?:to|for|assign(?:ed)?\s+to)\s+(?:me|myself)\b", text, re.I):
        meta["assignee_self"] = True

    # Assignee — Slack mention takes priority
    mention = _CREATE_MENTION.search(text)
    if mention:
        meta["assignee"] = f"<@{mention.group(1)}>"
    elif not meta.get("assignee_self"):
        named = _CREATE_ASSIGNEE.search(text)
        if named:
            name = named.group(1).strip()
            # Don't capture generic words as assignee names
            _NOT_NAMES = {
                "me", "i", "my", "us", "we", "you", "them",
                "today", "tomorrow", "yesterday", "monday", "tuesday", "wednesday",
                "thursday", "friday", "saturday", "sunday",
                "work", "do", "add", "create", "finish", "complete", "handle", "tackle", "deploy",
                "high", "medium", "low", "urgent", "critical", "normal",
            }
            if name.lower() not in _NOT_NAMES:
                meta["assignee"] = name

    return meta


def _clean_task_name(name: str) -> str:
    """Strip trailing metadata tokens and punctuation from a task name."""
    name = _TASK_META_SUFFIX.sub("", name)
    name = name.strip(" .,;:\\'\"")
    # Strip trailing noise words (task / item / action item)
    name = re.sub(r"\b(task|item|action\s+item|todo|to-do)s?\s*$", "", name, flags=re.I).strip()
    return name


def _clean_single_task_name(name: str, meta: dict) -> str:
    """
    Thoroughly strip metadata clauses (assignee, due date, priority)
    from a single-task CREATE description so the task name is clean.
    """
    # Remove priority clauses
    name = re.sub(r"\b(?:with\s+)?priority\s+p?[1-4]\b", "", name, flags=re.I)
    name = re.sub(r"\b(?:with\s+)?priority\s+(?:urgent|critical|highest|high|medium|normal|low|lowest)\b", "", name, flags=re.I)
    name = re.sub(r"\b(p[1-4])\b", "", name, flags=re.I)

    # Remove due date clauses
    name = re.sub(
        r"\b(?:due(?:\s+date)?|deadline|by)\s+(?:today|tomorrow|yesterday|this\s+week|next\s+\w+|\w+day|\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2}(?:,?\s+\d{4})?)\b",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(r"\b(?:due\s+date|due|deadline)\b", "", name, flags=re.I)

    # Remove assignee clauses ("to me", "for me", "to @user", "for @user", "assigned to @user")
    name = re.sub(r"\b(?:for|to|assign(?:ed)?\s+to)\s+(?:me|myself)\b", "", name, flags=re.I)
    if meta.get("assignee"):
        raw_assignee = meta["assignee"]
        if raw_assignee.startswith("<@") and raw_assignee.endswith(">"):
            name = name.replace(raw_assignee, "")
        else:
            clean_name = raw_assignee.lstrip("@")
            name = re.sub(rf"\b(?:for|to|assign(?:ed)?\s+to)\s+@?{re.escape(clean_name)}\b", "", name, flags=re.I)
            name = re.sub(rf"\b@?{re.escape(clean_name)}\b", "", name, flags=re.I)

    name = re.sub(r"\b(?:for|to|assign(?:ed)?\s+to)\s+<@[A-Z0-9]+(?:\|[^>]+)?>", "", name, flags=re.I)
    name = re.sub(r"<@[A-Z0-9]+(?:\|[^>]+)?>", "", name)

    # Strip dangling trailing prepositions / words: to, for, assigned, due, of, with, by, on, at
    name = re.sub(r"\s+\b(?:to|for|assigned\s+to|assigned|due|of|with|by|on|at)\s*$", "", name, flags=re.I).strip()
    # Strip dangling leading prepositions / words
    name = re.sub(r"^\b(?:to|for|assigned\s+to|assigned|due|of|with|by|on|at)\s+", "", name, flags=re.I).strip()

    name = _clean_task_name(name)
    return name


def _local_parse_create(text: str):
    """
    Deterministic parser for CREATE intents, with full multi-task support.

    Handles:
    - Bullet list lines  (* Task A, - Task B, • Task C)
    - Numbered list lines (1. Task A, 2. Task B)
    - Multiple sentences ("Finish X. Review Y. Send Z.")
    - Natural-language "task A and task B" with shared metadata
    - Single-task CREATE messages when they contain a clear CREATE phrase

    Returns:
      {"intent": "create", "tasks": [...], "raw_text": ...}  — for multi-task
      {"intent": "create", "task_name": "...", ...}          — for single-task
      {}                                                       — fall-through to Ollama
    """
    t = text.strip()

    # Never intercept mutation intents
    if (_DELETE_ANCHOR.match(t) or _UPDATE_ANCHOR.match(t)
            or _COMPLETE_PREFIX.match(t) or _COMPLETE_PASSIVE.match(t)
            or _REOPEN_ANCHOR.match(t)):
        return {}

    # Never intercept read intents
    if _READ_ANCHORS.match(t) or _WHAT_SHOULD.match(t) or _WHO_BARE.match(t):
        return {}

    lines = [l.strip() for l in t.splitlines() if l.strip()]

    # ── 1. Bullet / numbered list detection ──────────────────────────────────
    bullet_tasks = []
    numbered_tasks = []
    for line in lines:
        bm = _BULLET_LINE.match(line)
        if bm:
            name = _clean_task_name(bm.group(1))
            if name:
                bullet_tasks.append(name)
            continue
        nm = _NUMBERED_LINE.match(line)
        if nm:
            name = _clean_task_name(nm.group(1))
            if name:
                numbered_tasks.append(name)

    list_tasks = bullet_tasks or numbered_tasks
    has_list = bool(list_tasks)

    # Check for CREATE intent — required unless a list was detected
    has_create_intent = bool(_CREATE_ANCHOR.search(t))

    if not has_list and not has_create_intent:
        return {}

    # Shared metadata from the full message text
    meta = _extract_create_metadata(t)

    # Determine assignee_self: first-person CREATE ("I need to work on", "I want to", etc.)
    # We check against the FULL text (before verb stripping) so "I need to" is always matched.
    # Only set if no explicit third-party assignee was found
    _FIRST_PERSON_CREATE = re.compile(
        r"\b(i\s+(?:need|want|have|am|will|would)|my\s+(?:tasks?|action\s+items?))\b", re.I
    )
    if "assignee" not in meta and _FIRST_PERSON_CREATE.search(t):
        meta["assignee_self"] = True

    # ── 2. List-format result ─────────────────────────────────────────────────
    if has_list and len(list_tasks) >= 1:
        tasks = []
        for name in list_tasks:
            entry = {"task_name": name}
            entry.update(meta)  # Apply shared metadata
            tasks.append(entry)

        if len(tasks) == 1:
            # Single item in list — return as flat dict for backwards compat
            r = {"intent": "create", "raw_text": t, "task_name": tasks[0].pop("task_name")}
            r.update(tasks[0])
            return r

        return {"intent": "create", "tasks": tasks, "raw_text": t}

    # ── 3. Sentence-boundary multi-task ("Do X. Review Y. Send Z.") ──────────
    # Only attempt when there are 2+ lines or sentence splits
    full_text_for_split = t
    # Strip the CREATE verb prefix to get only the task body
    task_body = _CREATE_VERB_STRIP.sub("", full_text_for_split, count=1).strip()
    if not task_body:
        task_body = full_text_for_split

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(task_body) if s.strip()]
    sentence_tasks = []
    for s in sentences:
        # Skip lines that look like meta-commentary ("i want to work on", etc.)
        if _CREATE_VERB_STRIP.match(s) and len(sentences) > 1:
            continue
        name = _clean_task_name(s)
        if name and len(name.split()) >= 1:
            sentence_tasks.append(name)

    if len(sentence_tasks) >= 2:
        tasks = []
        for name in sentence_tasks:
            entry = {"task_name": name}
            entry.update(meta)
            tasks.append(entry)
        return {"intent": "create", "tasks": tasks, "raw_text": t}

    # ── 4. Natural-language "task A and task B" splitting ─────────────────────
    # Split on " and " only when:
    #  a) there is a clear CREATE intent
    #  b) both halves produce non-empty task names after stripping metadata
    #  c) neither half matches mutation/read patterns
    if has_create_intent and " and " in task_body.lower():
        # Split on " and " — but only the LAST occurrence to avoid splitting
        # task names that legitimately contain "and" (e.g. "Research and Development")
        # Strategy: split on " and " that is NOT followed by more " and " clauses
        # We do a simple 2-way split on the last " and " if both sides look like tasks.
        lower_body = task_body.lower()
        idx = lower_body.rfind(" and ")
        if idx != -1:
            left = task_body[:idx].strip()
            right = task_body[idx + 5:].strip()
            left_name = _clean_task_name(left)
            right_name = _clean_task_name(right)
            # Both parts must be non-empty and reasonable task names (>=2 chars, not just metadata)
            _META_ONLY = re.compile(
                r"^(?:p[1-4]|today|tomorrow|both|all|priority)$", re.I
            )
            if (left_name and right_name
                    and len(left_name) >= 2 and len(right_name) >= 2
                    and not _META_ONLY.fullmatch(left_name)
                    and not _META_ONLY.fullmatch(right_name)):
                tasks = []
                for name in (left_name, right_name):
                    entry = {"task_name": name}
                    entry.update(meta)
                    tasks.append(entry)
                return {"intent": "create", "tasks": tasks, "raw_text": t}

    # ── 5. Single-task CREATE ─────────────────────────────────────────────────
    # Strip the verb prefix and return a flat single-task dict with cleaned task name
    name = _clean_single_task_name(task_body, meta)
    if not name:
        # Could not extract a task name — fall through to Ollama
        return {}

    r: dict = {"intent": "create", "raw_text": t, "task_name": name}
    r.update(meta)
    return r


# ---------------------------------------------------------------------------
# Deterministic local READ parser — regex patterns
# ---------------------------------------------------------------------------

# Mutations — never intercept in the read parser
_READ_ANCHORS = re.compile(
    r"^(list|show|get|display|fetch|what|which|who|how\s+many|tell\s+me|give\s+me|find|anything|is\s+there|are\s+there|check|view|see)\b",
    re.I,
)
_WHAT_SHOULD = re.compile(
    r"^what\s+(should|can|must|do|does|will|would|am|are|is|have|'s)\b",
    re.I,
)
_WHO_BARE = re.compile(r"^who\b", re.I)

# ── ASSIGNEE ──────────────────────────────────────────────────────────────
# Explicit Slack mention <@UXXXXXX|name> or <@UXXXXXX>
_MENTION_ASSIGNEE = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]+)?>", re.I)

# Named third-person after prepositions or "@Name" (bare at-sign)
_NAMED_ASSIGNEE = re.compile(
    r"\b(?:assigned\s+to|tasks?\s+of|work(?:ing)?\s+on\s+by|by|@)([A-Za-z]\w+)\b",
    re.I,
)
# Don't let _NAMED_ASSIGNEE match first-person words
_FIRST_PERSON_WORDS = {"me", "i", "my", "us", "we", "our", "myself"}

# First-person pronoun anywhere in sentence (only set assignee_self when no third-party found)
_FIRST_PERSON = re.compile(r"\b(i|me|my)\b", re.I)

# Strong self-assignee anchoring phrases
_SELF_ASSIGNEE_PHRASE = re.compile(
    r"\b("
    r"my\s+(tasks?|action\s+items?|work|todo|next|nearest|deadline|plate)|"
    r"assigned\s+to\s+me|"
    r"what\s+(do\s+i|i\s+have|should\s+i|i\s+need\s+to|am\s+i)|"
    r"what\s+i\s+(have|need|should|must|can|will)|"
    r"i\s+have\s+to\s+(work|do|finish|complete)|"
    r"i\s+need\s+to\s+(work|do|finish|complete)|"
    r"should\s+i\s+(work|do|focus|tackle)"
    r")\b",
    re.I,
)

# ── FILTERS ───────────────────────────────────────────────────────────────
_PRIORITY = re.compile(r"\b(p[1-4])\b", re.I)
_STATUS_COMPLETED = re.compile(r"\b(completed?|done|finish(?:ed|ing)?|closed)\b", re.I)
_STATUS_OPEN = re.compile(r"\b(pending|open|incomplete|not\s+done|not\s+completed?|in\s+progress|left|remaining|to\s+do)\b", re.I)

_DUE_TODAY = re.compile(
    r"\b(today|due\s+today|for\s+today|this\s+day)\b",
    re.I,
)
_OVERDUE = re.compile(r"\b(overdue|past\s+due|late|missed\s+deadline)\b", re.I)
_DUE_WEEK = re.compile(r"\b(this\s+week|this\s+week'?s?|week)\b", re.I)

_ALL_TASKS = re.compile(
    r"\b(all\s+(?:my\s+)?(?:tasks?|action\s+items?|everything)|every\s+(task|action\s+item)|"
    r"show\s+everything|list\s+everything)\b",
    re.I,
)
_BARE_LIST = re.compile(r"^(list|show|get)(\s+all)?(\s+my)?\s+(tasks?|action\s+items?)\s*$", re.I)

# ── SORT / LIMIT ──────────────────────────────────────────────────────────
_SORT_NEAREST = re.compile(
    r"\b(nearest|closest|earliest|soonest|upcoming)\s+(deadline|due|task|item)\b"
    r"|\b(due\s+soonest|soonest\s+due|earliest\s+due|nearest\s+deadline)\b"
    r"|\bnext\s+deadline\b",
    re.I,
)
_NEXT_TASK = re.compile(
    r"\b(work\s+on\s+next|do\s+next|focus\s+on\s+next|tackle\s+next|"
    r"my\s+next\s+task|next\s+task|next\s+item|should\s+i\s+work\s+on\s+next|"
    r"should\s+i\s+do\s+next|should\s+i\s+focus\s+on\s+next)\b",
    re.I,
)
_LIMIT_ONE = re.compile(
    r"\b(one\s+(task|item|action\s+item)|a\s+single\s+(task|item)|single\s+(task|item))\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Deterministic local parser — entry point
# ---------------------------------------------------------------------------

def _local_parse(text: str) -> Dict[str, Any]:
    """
    Deterministic parser for clear READ commands only.
    Returns {} to fall through for mutations or ambiguous inputs.
    """
    t = text.strip()

    # 1. Never intercept mutations — they are handled by _local_parse_mutation
    if (_DELETE_ANCHOR.match(t) or _UPDATE_ANCHOR.match(t)
            or _COMPLETE_PREFIX.match(t) or _COMPLETE_PASSIVE.match(t)
            or _REOPEN_ANCHOR.match(t)):
        return {}

    # 2. Must start with a recognised read anchor
    if not (_READ_ANCHORS.match(t) or _WHAT_SHOULD.match(t) or _WHO_BARE.match(t)):
        return {}

    res: Dict[str, Any] = {"intent": "list"}

    # ── Date / status filters ──
    if _DUE_TODAY.search(t):
        res["due_today"] = True
    if _OVERDUE.search(t):
        res["overdue"] = True
    if _DUE_WEEK.search(t) and not res.get("due_today") and not res.get("overdue"):
        res["due_this_week"] = True
    if _STATUS_COMPLETED.search(t):
        res["status"] = "completed"
        res["completed"] = True
    elif _STATUS_OPEN.search(t):
        res["status"] = "open"
        res["completed"] = False
    pm = _PRIORITY.search(t)
    if pm:
        res["priority"] = pm.group(1).upper()

    # ── Sort / limit dimensions ──
    if _SORT_NEAREST.search(t) or _NEXT_TASK.search(t):
        res["sort_by"] = "due_date"
        res["sort_order"] = "asc"
        res["limit"] = 1
        if "completed" not in res:
            res["completed"] = False

    if _LIMIT_ONE.search(t) and "limit" not in res:
        res["limit"] = 1
        res["sort_by"] = "due_date"
        res["sort_order"] = "asc"
        if "completed" not in res:
            res["completed"] = False

    # ── Assignee dimension ──
    # Priority: explicit Slack mention > named third-person > first-person self
    mention = _MENTION_ASSIGNEE.search(t)
    if mention:
        res["assignee"] = f"<@{mention.group(1)}>"
    else:
        named = _NAMED_ASSIGNEE.search(t)
        if named and named.group(1).lower() not in _FIRST_PERSON_WORDS:
            res["assignee"] = named.group(1)

        # First-person: set assignee_self when no third-party assignee is present
        # This applies for ALL query types, including due_today combinations.
        if not res.get("assignee") and (
            _SELF_ASSIGNEE_PHRASE.search(t) or _FIRST_PERSON.search(t)
        ):
            res["assignee_self"] = True

    # ── "All tasks" bare list ──
    is_all = bool(_ALL_TASKS.search(t))
    is_generic = bool(_BARE_LIST.match(t.lower()) or is_all or t.lower() in ("list", "show"))
    is_who = bool(_WHO_BARE.match(t))

    if is_all:
        res["all_tasks"] = True
        res["completed"] = None
        res.pop("status", None)

    has_filter = any(k in res for k in (
        "due_today", "overdue", "due_this_week", "status", "priority",
        "assignee", "assignee_self", "sort_by", "limit", "completed", "all_tasks",
    ))

    if is_generic or is_who or has_filter:
        if not is_all and "status" not in res and "completed" not in res:
            res["status"] = "open"
            res["completed"] = False
        return res

    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_intent(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    # Strip leading Slack bot mention
    text = re.sub(r"^\s*<@[A-Z0-9]+(?:\|[^>]+)?>\s*", "", text, flags=re.I)
    # Strip trailing bot trigger mention if not preceded by assignment preposition
    m = re.search(r"\s*<@[A-Z0-9]+(?:\|[^>]+)?>\s*$", text, flags=re.I)
    if m:
        prefix = text[:m.start()].rstrip()
        if not re.search(r"\b(?:to|for|by|assign|assigned\s+to)\s*$", prefix, flags=re.I):
            text = prefix

    text = re.sub(r"^\s*/add\b\s*", "", text, flags=re.I).strip()

    if not text:
        return _empty_result()

    # ── 1. Deterministic mutation parser (no Ollama needed) ──────────────────
    local_mut = _local_parse_mutation(text)
    if local_mut:
        result = _empty_result(text)
        result.update(local_mut)
        return result

    # ── 2. Deterministic read parser ─────────────────────────────────────────
    local_read = _local_parse(text)
    if local_read:
        result = _empty_result(text)
        result.update(local_read)
        return result

    # ── 2b. Deterministic CREATE parser (multi-task support) ─────────────────
    local_create = _local_parse_create(text)
    if local_create:
        result = _empty_result(text)
        result.update(local_create)
        return result

    # ── 3. Ollama fallback (ambiguous / complex commands) ────────────────────
    if not OLLAMA_API_KEY:
        logger.error("OLLAMA_API_KEY is missing — cannot call Ollama")
        return _empty_result(text)

    # Safe date injection — str.replace leaves JSON braces untouched
    prompt = _inject_dates(SYSTEM_PROMPT)
    llm = ChatOllama(model=OLLAMA_MODEL, format="json", temperature=0.0)

    # Max 2 retries — only for genuine transient transport errors (empty body, connection reset).
    # JSON parse failures are NOT retried — log and return out_of_scope.
    for attempt in range(3):
        content = ""
        try:
            messages = [SystemMessage(content=prompt), HumanMessage(content=text)]
            response = llm.invoke(messages)
            content  = (response.content or "").strip()

            if not content:
                # Transient empty body — retry with backoff
                if attempt < 2:
                    delay = 2 ** attempt
                    logger.warning("Ollama returned empty body, retry %d in %ds", attempt + 1, delay)
                    time.sleep(delay)
                    continue
                logger.error("Ollama returned empty body after %d attempts for: %s", attempt + 1, text)
                return {"intent": "temporarily_unavailable"}

            # Strip accidental markdown fences
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```\s*$", "", content)

            parsed = json.loads(content)
            result = _empty_result(text)
            result.update(parsed)
            # Normalise top-level due_date
            if result.get("due_date") in ("today", "TODAY"):
                result["due_date"] = _today()
            elif result.get("due_date") in ("tomorrow", "TOMORROW"):
                result["due_date"] = _tomorrow()
            # Normalise due_date inside each task entry (multi-task path)
            for task_entry in result.get("tasks") or []:
                if not isinstance(task_entry, dict):
                    continue
                if task_entry.get("due_date") in ("today", "TODAY"):
                    task_entry["due_date"] = _today()
                elif task_entry.get("due_date") in ("tomorrow", "TOMORROW"):
                    task_entry["due_date"] = _tomorrow()
            return result

        except json.JSONDecodeError as exc:
            # JSON parse failure — model returned garbled output. Don't retry.
            logger.error(
                "Ollama returned invalid JSON for %r — snippet: %r — error: %s",
                text, content[:200], exc,
            )
            return _empty_result(text)

        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                logger.warning("Ollama rate limit hit — not retrying")
                return {"intent": "temporarily_unavailable"}
            if attempt < 2:
                delay = 2 ** attempt
                logger.warning("Ollama transport error, retry %d in %ds: %s", attempt + 1, delay, exc)
                time.sleep(delay)
            else:
                logger.exception("Ollama failed after 3 attempts for: %s", text)
                return {"intent": "temporarily_unavailable"}

    return {"intent": "temporarily_unavailable"}


# Backwards-compatible aliases
def parse(text: str) -> Dict[str, Any]:
    return parse_intent(text)

def parse_request(text: str) -> Dict[str, Any]:
    return parse_intent(text)

def parse_command(text: str) -> Dict[str, Any]:
    return parse_intent(text)
