import os
import re
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from references import Reference, parse_reference, collection_scope, extract_contextual_reference, reference_from
from dataclasses import asdict
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

    if re.search(r"\bnext\s+week\b", t):
        return (today + timedelta(days=7-today.weekday())).isoformat()

    # Weekdays calculation
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for name, wday in weekdays.items():
        if re.search(rf"\bnext\s+{name}\b", t):
            days_ahead = (wday - today.weekday()) % 7 or 7
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
            for fmt in ("%B %d, %Y", "%B %d %Y", "%d %B %Y", "%b %d %Y", "%d %b %Y"):
                try:
                    d = datetime.strptime(raw.title().replace(",", ""), fmt.replace(",", "")).date()
                    return d.isoformat()
                except ValueError:
                    pass
        except Exception:
            pass
    # A month/day without a year means its next non-past occurrence.
    m = re.search(rf"\b({months})\s+(\d{{1,2}})\b", t, re.I)
    if m:
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                candidate = datetime.strptime(f"{m.group(0).title()} {today.year}", fmt).date()
                if candidate < today:
                    candidate = candidate.replace(year=today.year + 1)
                return candidate.isoformat()
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Standardized Output Schema
# ---------------------------------------------------------------------------

def _empty_result(raw_text: str = "") -> Dict[str, Any]:
    return {
        "intent": "out_of_scope",
        "tasks": [],
        "operations": [],
        "task_name": None,
        "task_reference": None,
        "selection": None,
        "selection_index": None,
        "priority": None,
        "status": None,
        "status_filter": None,
        "statuses": [],
        "assignee": None,
        "assignees": [],
        "assignee_reference": None,
        "assignee_self": False,
        "member": None,
        "members": [],
        "member_self": False,
        "role": None,
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

ALLOWED INTENTS: create | list | inspect | progress | health | plan | workload | standup | apply_proposal | confirm | cancel | history | dependencies | update | complete | reopen | delete | members | compound | out_of_scope | clarify

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
   Built-in fields: assignee, priority, due_date, status, name. Additional
   configured Slack List fields may use their exact schema key or column name;
   authorization and schema validation happen after parsing.
   "to P2" / "priority to P3" -> changes=[{"field":"priority","value":"P2"}]
8. "my tasks" / first-person "I"/"me"/"my" in a LIST -> assignee_self=true.
   For delete/update/create, "my" means the task is assigned to the requester.
9. List filters: due_today (bool), overdue (bool), due_this_week (bool), completed (bool), query (str),
   date_from/date_to (inclusive YYYY-MM-DD bounds).
10. Pronouns ("this", "that", "the first one") -> task_name: "__LAST__", selection: "first"/"both"/"all".
11. If genuinely ambiguous -> intent: "clarify", question in "clarification".
12. General task queries -> intent "list" with appropriate filters.
    Specific task information/status/assignee questions -> intent "inspect", task_name or reference.
    Never interpret a question about completion as a completion command.
    Scope: unrelated factual/advice/programming/entertainment questions are out_of_scope, even if they contain a word such as "list", "my", "work", or "first".
    Never invent a task from an unrelated question.
    Reference noun phrases belong in task_name (e.g. "sixth one", "it", "both").
    Actual titles containing reference words remain titles. Quoted titles set literal_name=true.
    Separate assignee (target filter) from changes[field=assignee] (new assignee).
    For one or more target-user filters, use assignees=[exact user text...].
    Multiple assignees use OR semantics. A reassignment uses changes[field=assignee],
    whose value is one user or a list of users. Never place the new owner in the
    target-filter assignee/assignees fields.
    References such as "their tasks", "both users", or "these users" use
    assignee_reference="context"; do not invent names.
    Missing field values require clarify; never invent them.
13. assignee: exact text the user typed (display name or @mention). Do NOT invent user IDs.
14. "nearest deadline" / "next deadline" -> sort_by="due_date", sort_order="asc", limit=1.
15. "one task" / "single task" -> limit=1, sort_by="due_date", sort_order="asc".
16. "what should I work on next?" -> assignee_self=true, sort_by="due_date", sort_order="asc", limit=1, completed=false.
17. MULTIPLE TASKS: If the user provides multiple tasks in one message (bullet list, numbered list, or "task A and task B"):
    - Preserve the requested intent; use create only if the user requests creation.
    - Use "tasks" list (NOT task_name) with one entry per task.
    - Each entry: {"task_name": "...", "assignee": "...", "assignee_self": true/false, "priority": "...", "due_date": "YYYY-MM-DD"}.
    - Shared metadata (assignee, due_date, priority) applies to ALL entries unless overridden per task.
    - First-person ("I", "me", "my", "I want to work on") -> assignee_self=true for each task entry.
    - Omit null/false/missing fields inside each task entry.
18. MULTIPLE OPERATIONS: If one message requests two or more distinct actions, use
    intent="compound" and operations=[...] with one complete ordinary command per action,
    in requested order. Do not use compound merely because one action has multiple targets.
    Never nest compound operations.
19. WORKSPACE MEMBERS: Questions that ask who workspace members are, identify a
    member, or ask about configured application roles use intent="members".
    Preserve typed names or mentions in members=[...]. Use member_self=true for
    the requesting user. A role filter uses role="admin|manager|member|viewer".
    Do not turn member or role questions into task queries.
20. TARGET SCOPE: Represent target meaning independently from wording. Use
    target_scope="all_applicable" for every current Slack List item,
    "filtered" for every item matching filters, "contextual" for items from a
    prior display, "multiple" for explicitly named multiple items, and "single"
    for one named item. Bare contextual words such as "all of those" refer to
    the displayed set; an explicit collection such as all tasks refers to live
    applicable Slack List data.
21. ANALYTICAL QUERIES: Preserve the operation to perform after retrieval.
    Use sort_by=name|due_date|priority|status|assignee and sort_order=asc|desc;
    group_by=assignee|status|priority|due_date; aggregate="count" for counts;
    and limit for requested quantities. Comparisons of category counts are
    group_by plus aggregate="count". These fields compose with every filter.
22. TARGET SELECTION: Keep candidate filters separate from the operation that
    selects records. target_selection is {"mode":"one|many|collection",
    "order_by":"position|created_at|due_date|priority|name|status|assignee",
    "direction":"asc|desc", "count":1}. A superlative or singular positional
    request uses mode="one". Filters never imply collection output.
23. RESULT OPERATION: use result_operation=return_collection|select_one|
    select_many|aggregate|comparison|summary. "Which task" and qualitative
    superlatives select one; they do not return the candidate collection.
24. TEMPORAL DIMENSIONS: temporal_filter separates the field being compared:
    field=due_date|completed_at|created_at|updated_at, relation=on|before|after|
    between, and date/date_from/date_to. "completed today" uses completed_at;
    it never uses due_today. Do not infer unavailable historical facts.
25. PROGRESS AND INSIGHTS: use intent="progress" for progress, workload,
    completion-rate, distribution, risk, deadline-summary, or time-series
    questions. Use analytics_metrics with values overview, completion, workload,
    status_distribution, priority_distribution, overdue, due_today,
    due_this_week, upcoming, at_risk, completed_over_time, created_over_time,
    comparison, or summary. Use analytics_period={"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}
    for a requested reporting period. Never substitute due dates for completion
    or creation timestamps.
26. PROJECT INTELLIGENCE: use health for factual task-health/risk explanations;
    plan for a proposed schedule; workload for overload or rebalancing analysis;
    standup for daily standups; history for mutation audit questions; and
    dependencies for explicit blocker/dependency questions. A proposal never
    mutates tasks. apply_proposal requires a prior proposal in this thread.
    confirm/cancel act only on a prior pending confirmation.

JSON SCHEMA (omit null/false/missing fields):
{
  "intent": "create",
  "operations": [],
  "task_name": "...",
  "tasks": [{"task_name": "...", "assignee": "member name or mention", "assignee_self": true, "priority": "P1", "due_date": "YYYY-MM-DD"}],
  "priority": "P1",
  "status": "open",
  "assignee": "member name or mention",
  "assignees": ["member one", "member two"],
  "assignee_reference": "context",
  "assignee_self": true,
  "members": ["Alex"],
  "member_self": false,
  "role": "manager",
  "due_date": "YYYY-MM-DD",
  "completed": false,
  "query": "search term",
  "due_today": false,
  "overdue": false,
  "due_this_week": false,
  "date_from": "YYYY-MM-DD",
  "date_to": "YYYY-MM-DD",
  "selection": "first",
  "selection_count": 4,
  "target_scope": "filtered",
  "limit": 1,
  "sort_by": "due_date",
  "sort_order": "asc",
  "group_by": "assignee",
  "aggregate": "count",
  "count_only": false,
  "target_selection": {"mode": "one", "order_by": "due_date", "direction": "asc", "count": 1},
  "result_operation": "select_one",
  "temporal_filter": {"field": "completed_at", "relation": "on", "date": "YYYY-MM-DD"},
  "analytics_metrics": ["overview", "workload"],
  "analytics_period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "analytics_comparison": {"current": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}, "previous": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},
  "planning_period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "attention_only": true,
  "recommend_balance": true,
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
    return name


def _build_mutation_result(intent, task_name, text, body, t, changes=None, assignee_raw=None, assignee_self=False):
    r = _empty_result(text)
    r["intent"] = intent

    # Recognize a reference only when the entire target is a reference phrase.
    quoted = _QUOTED.search(body or "")
    target_text = task_name if task_name != "__LAST__" else body
    # Use the original target clause before title cleanup removes collection nouns.
    set_info = None if quoted else (_parse_set_reference(body) or _parse_set_reference(target_text))
    ref = set_info["reference"] if set_info else (None if quoted else parse_reference(target_text))
    if ref:
        r["reference"] = asdict(ref)
        r["task_name"] = "__LAST__"
        r["selection"] = ref.kind
        if ref.kind == "positions" and len(ref.positions) == 1:
            r["selection_index"] = ref.positions[0]
    elif task_name:
        r["task_name"] = task_name
        r["literal_name"] = bool(quoted)

    if changes:
        r["changes"] = changes
    if assignee_raw:
        r["assignee"] = assignee_raw
    if assignee_self:
        r["assignee_self"] = True
    if set_info:
        r.update({k: v for k, v in set_info.items() if k != "reference"})
        r["reference_scope"] = "filtered"
    return r


def _parse_set_reference(text):
    """Parse a broad, explicitly filtered set separately from a displayed set."""
    value = str(text or "").strip().casefold().rstrip(".!?")
    scope = collection_scope(value)
    if not scope and not re.search(r"\b(?:all|everything|both)\b", value):
        return None
    scoped = bool(re.search(
        r"\b(?:my|mine|assigned to me|on my list|pending|open|outstanding|completed|done|overdue|due today)\b",
        value,
    ))
    all_applicable = scope == "all_applicable"
    scoped = scoped or all_applicable
    if not scoped:
        return None
    result = {"reference": Reference("both" if re.search(r"\bboth\b", value) else "all")}
    result["target_scope"] = "all_applicable" if all_applicable else "filtered"
    if re.search(r"\b(?:my|mine|assigned to me|on my list)\b", value):
        result["assignee_self"] = True
    if re.search(r"\b(?:pending|open|outstanding|unfinished|still)\b", value):
        result["completed"] = False
    elif re.search(r"\b(?:completed|done|finished)\b", value):
        result["completed"] = True
    if re.search(r"\boverdue\b", value):
        result["overdue"] = True
    if re.search(r"\bdue today\b", value):
        result["due_today"] = True
    return result


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
    if is_complete and " and " in t.lower() and not passive_task and not parse_reference(_COMPLETE_PREFIX.sub("", t, count=1).strip()):
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
                "tasks": [_build_mutation_result("complete", p, text, p, p) for p in cleaned_parts],
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
        task_name = (q.group(1).strip() if q else _clean_task(body))
        if not task_name:
            return {}
        return _build(verb, task_name)

    # ── delete ──
    if verb == "delete":
        q = _QUOTED.search(body)
        task_name = (q.group(1).strip() if q else _clean_task(body))
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
    task_name = (q.group(1).strip() if q else _clean_task(task_body))

    if not task_name and not changes:
        return {}

    if not changes:
        if re.search(r"\b(priority|deadline|due date|assignee|status)\s*$", body, re.I):
            return {"intent": "clarify", "clarification": "Please specify the new field value."}
        return {}
    return _build("update", task_name, changes)


# ---------------------------------------------------------------------------
# Deterministic local CREATE parser — multi-task support
# ---------------------------------------------------------------------------

# CREATE intent anchor phrases (must appear in the message OR message has bullet/numbered lines)
_CREATE_ANCHOR = re.compile(
    r"\b("
    r"add|create|assign|i\s+have\s+(?:action\s+items?|tasks?|todo)|i\s+want\s+to\s+(?:work|add|create)|"
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
    r"(?:please\s+)?(?:add|create|assign)\s+"
    r"(?:(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+|some|several|a\s+couple\s+of|a\s+pair\s+of)\s+)?"
    r"(?:(?:action\s+)?(?:items?|tasks?)\s*)?(?:called\s+|named\s+|of\s+)?|\s*"
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


def _extract_create_due_clause(text: str):
    """Return ``(normalized_date, span)`` for a creation date clause.

    Generic ``for`` syntax is accepted only when its value resolves as a date,
    so ordinary task titles and member relationships remain untouched.
    """
    boundary = r"(?=\s+(?:for|to|assign(?:ed)?\s+to|priority|p[1-4])\b|[,;]|$)"
    patterns = (
        rf"\b(?:due(?:\s+date)?|deadline|by|to\s+date)\s+(.+?){boundary}",
        rf"\bfor\s+(.+?){boundary}",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            normalized = _resolve_natural_date(match.group(1).strip())
            if normalized:
                return normalized, match.span()
    return None


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

    # Date clauses are resolved before generic for/to assignee grammar.
    due_m = _extract_create_due_clause(text)
    if due_m:
        meta["due_date"] = due_m[0]
    else:
        # Preserve an explicit but invalid value so deterministic validation
        # rejects it rather than silently creating an undated task.
        invalid_due = re.search(
            r"\b(?:due(?:\s+date)?|deadline|by|to\s+date)\s+"
            r"(.+?)(?=\s+(?:for|to|assign(?:ed)?\s+to|priority|p[1-4])\b|[,;]|$)",
            text, re.I)
        if invalid_due:
            meta["due_date"] = invalid_due.group(1).strip()

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
        assignee_text = text
        if due_m:
            assignee_text = text[:due_m[1][0]] + " " + text[due_m[1][1]:]
        named = _CREATE_ASSIGNEE.search(assignee_text)
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

    # Use the same recognized span for field extraction and title cleanup.
    due_clause = _extract_create_due_clause(name)
    if due_clause:
        start, end = due_clause[1]
        name = name[:start] + " " + name[end:]

    # Remove remaining explicit due-date syntax for invalid-date validation.
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

    name = re.sub(r"\s+and\s*$", "", name, flags=re.I)
    name = re.sub(r"\s+", " ", name)
    name = _clean_task_name(name)
    return name


def _create_entry(text, shared):
    meta = dict(shared)
    own = _extract_create_metadata(text)
    # A generic title such as "Send to client" does not assign a Slack member.
    if not (re.search(r"<@|@|\bassigned?\s+to\s+", text, re.I) or re.search(r"\bfor\s+(?:[A-Z]|me\b|myself\b)", text)):
        own.pop("assignee", None)
        own.pop("assignee_self", None)
    if own.get("assignee"):
        meta.pop("assignee_self", None)
    if own.get("assignee_self"):
        meta.pop("assignee", None)
    meta.update(own)
    name = _clean_single_task_name(text, own)
    return {"task_name": name, **meta}


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
            name = bm.group(1).strip()
            if name:
                bullet_tasks.append(name)
            continue
        nm = _NUMBERED_LINE.match(line)
        if nm:
            name = nm.group(1).strip()
            if name:
                numbered_tasks.append(name)

    list_tasks = bullet_tasks or numbered_tasks
    has_list = bool(list_tasks)

    # Check for CREATE intent — required unless a list was detected
    has_create_intent = bool(_CREATE_ANCHOR.search(t))

    if not has_list and not has_create_intent:
        return {}

    # List metadata belongs to the header, never to the first task by accident.
    header = " ".join(line for line in lines if not _BULLET_LINE.match(line) and not _NUMBERED_LINE.match(line))
    meta = _extract_create_metadata(header if has_list else t)

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
            tasks.append(_create_entry(name, meta))

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
            tasks.append(_create_entry(name, meta))
        return {"intent": "create", "tasks": tasks, "raw_text": t}

    # ── 4. Natural-language "task A and task B" splitting ─────────────────────
    # Split on " and " only when:
    #  a) there is a clear CREATE intent
    #  b) both halves produce non-empty task names after stripping metadata
    #  c) neither half matches mutation/read patterns
    if has_create_intent and " and " in task_body.lower() and not re.search(r"[\"“”]|\b(?:called|named)\b", t, re.I):
        parts = re.split(r"\s+and\s+", task_body, flags=re.I)
        meta_only = re.compile(r"^(?:(?:both|all)\s+)?(?:p[1-4]|priority\s+p[1-4]|today|tomorrow)[.!?]*$", re.I)
        while parts and meta_only.fullmatch(parts[-1].strip()):
            parts.pop()
        if len(parts) > 1:
            return {"intent": "create", "tasks": [_create_entry(part, meta) for part in parts], "raw_text": t}

    # ── 5. Single-task CREATE ─────────────────────────────────────────────────
    # Strip the verb prefix and return a flat single-task dict with cleaned task name
    quoted = _QUOTED.search(task_body)
    if quoted:
        name = quoted.group(1)
        meta = _extract_create_metadata(task_body[:quoted.start()] + task_body[quoted.end():])
    else:
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
    r"\b(?:assigned\s+to|tasks?\s+of|work(?:ing)?\s+on\s+by|by|@)\s*([A-Za-z]\w+)\b",
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
_STATUS_COMPLETED = re.compile(r"\b(completed?|done|finished|closed)\b|\bdid\s+(?:i\s+)?finish\b", re.I)
_STATUS_OPEN = re.compile(r"\b(pending|open|incomplete|not\s+done|not\s+completed?|in\s+progress|left|remaining|to\s+do)\b", re.I)

_DUE_TODAY = re.compile(
    r"\b(today|due\s+today|for\s+today|this\s+day)\b",
    re.I,
)
_OVERDUE = re.compile(r"\b(overdue|past\s+due|late|missed\s+deadline)\b", re.I)
_DUE_WEEK = re.compile(r"\b(this\s+week|this\s+week'?s?|week)\b", re.I)


def _priority_filter(text):
    explicit = _PRIORITY.search(text)
    if explicit:
        return explicit.group(1).upper()
    qualitative = re.search(
        r"\b(?:(urgent|critical)|((?:high|medium|normal|low)[-\s]+priority))\b",
        text, re.I)
    if qualitative:
        from config import normalize_priority
        return normalize_priority(qualitative.group(1) or re.split(r"[-\s]+", qualitative.group(2))[0])
    return None

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
            or _COMPLETE_PREFIX.match(t) or (_COMPLETE_PASSIVE.match(t) and not _READ_ANCHORS.match(t))
            or _REOPEN_ANCHOR.match(t)):
        return {}

    # 2. Must start with a recognised read anchor
    if not (_READ_ANCHORS.match(t) or _WHAT_SHOULD.match(t) or _WHO_BARE.match(t)):
        return {}

    domain = re.search(r"\b(tasks?|items?|action items?|todo|pending|overdue|deadlines?|assigned|completed|complete|finished|finish|left|remaining|work|due|open)\b", t, re.I)
    generic = re.fullmatch(r"(?:show|list)(?:\s+(?:me\s+)?(?:all|everything))?[.!?]*", t, re.I)
    self_work = re.fullmatch(r"what (?:should i do|do i (?:still )?(?:need|have) to do|do i still have)(?: today| next)?[.!?]*", t, re.I)
    if not domain and not generic and not self_work:
        return {}
    res: Dict[str, Any] = {"intent": "list"}
    if re.match(r"how many\b", t, re.I):
        res["count_only"] = True

    # ── Date / status filters ──
    if _DUE_TODAY.search(t):
        res["due_today"] = True
    if _OVERDUE.search(t):
        res["overdue"] = True
    if _DUE_WEEK.search(t) and not res.get("due_today") and not res.get("overdue"):
        res["due_this_week"] = True
    if _STATUS_OPEN.search(t) or re.search(r"\b(still|have to|need to)\b", t, re.I):
        res["status"] = "open"
        res["completed"] = False
    elif _STATUS_COMPLETED.search(t):
        res["status"] = "completed"
        res["completed"] = True
    priority = _priority_filter(t)
    if priority:
        res["priority"] = priority

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
    is_all = bool(_ALL_TASKS.search(t) or re.fullmatch(r"(?:show|list)(?:\s+me)?\s+all[.!?]*", t, re.I))
    is_generic = bool(_BARE_LIST.match(t.lower()) or is_all or t.lower() in ("list", "show"))
    is_who = bool(_WHO_BARE.match(t))

    if is_all:
        res["all_tasks"] = True
        if not (_STATUS_OPEN.search(t) or _STATUS_COMPLETED.search(t)):
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

def _target_result(intent, target, changes=None):
    target = target.strip().rstrip(".?!").strip()
    quoted = len(target) > 1 and target[0] in '\"“' and target[-1] in '\"”'
    qualified_member = None
    qualified_name = None
    ref = None if quoted else parse_reference(target)
    if not quoted and not ref:
        # A member qualifier and a complete reference are separate entities.
        # Only accept this split when the remainder is wholly reference grammar,
        # so ordinary task titles containing a person's name stay literal.
        qualified = re.fullmatch(r"(<@[A-Z0-9]+(?:\|[^>]+)?>|@[A-Za-z][\w.-]*)\s+(.+)", target, re.I)
        if qualified:
            candidate = parse_reference(qualified.group(2))
            qualified_member = qualified.group(1)
            if candidate:
                ref = candidate
            else:
                # Slack mrkdwn may wrap the spoken target in emphasis markers.
                # Remove only balanced surrounding markup, preserving the text.
                qualified_target = qualified.group(2).strip()
                if (len(qualified_target) >= 2 and qualified_target[0] == qualified_target[-1]
                        and qualified_target[0] in "*_~`"):
                    qualified_target = qualified_target[1:-1].strip()
                qualified_name = re.sub(r"\s+(?:task|item|entry)$", "", qualified_target, flags=re.I).strip()
    result = {"intent": intent, "task_name": qualified_name or target.strip('\"“”'), "literal_name": quoted}
    if ref:
        result.update(task_name="__LAST__", reference=asdict(ref))
        if qualified_member:
            result.update(assignees=[qualified_member], target_scope="filtered", reference_scope="filtered")
    elif qualified_member:
        result["assignees"] = [qualified_member]
    if changes:
        result["changes"] = changes
    return result


_PERSON_WORD = r"(?!(?:and|or|with|whose|that|which|above|below|displayed|previous|next|first|second|third|last|assign|reassign|move|give|transfer|complete|finish|reopen|delete|remove|update|change|show|list)\b)[A-Za-z][\w.-]*"
_PERSON_TOKEN = rf"(?:<@[A-Z0-9]+(?:\|[^>]+)?>|@[A-Za-z][\w.-]*|{_PERSON_WORD}(?:\s+{_PERSON_WORD})?)"


def _person_list(value):
    """Extract a coordinated list of user references without resolving identity."""
    value = re.sub(
        r"\b(?:whose|that|which|with|due|priority|pending|open|completed|overdue|today|this week)\b.*$",
        "", str(value or ""), flags=re.I,
    ).strip(" ,.;")
    if re.fullmatch(r"(?:them|their|both users|these users|those users|the two users)", value, re.I):
        return [], "context"
    parts = re.split(r"\s*(?:,|\band\b|&)\s*", value, flags=re.I)
    people = []
    for part in parts:
        part = re.sub(r"^(?:(?:show|list|find|display)\s+)?(?:both|user|users)?\s*", "", part.strip(), flags=re.I)
        part = re.sub(r"^(?:of|the)\s+", "", part, flags=re.I)
        if re.fullmatch(_PERSON_TOKEN, part, re.I):
            people.append(part)
    return list(dict.fromkeys(people)), None


def _assignee_filter(text):
    """Return target-user filters from possessive or relationship syntax."""
    text = text.strip().rstrip(".?!")
    if re.search(r"\b(?:unassigned|without\s+(?:an\s+)?assignee|with\s+no\s+(?:assignee|owner)|assigned\s+to\s+no\s+one)\b", text, re.I):
        return {"assignee_condition": "unassigned"}
    if re.search(
            r"\b(?:someone\s+else|somebody\s+else|another\s+(?:person|member|user)|"
            r"(?:someone|somebody|anyone|anybody)\s+other\s+than\s+(?:me|myself)|"
            r"not\s+assigned\s+to\s+(?:me|myself))\b", text, re.I):
        return {"assignee_condition": "other"}
    if re.search(r"\b(?:with\s+any\s+assignee|assigned\s+to\s+(?:anyone|anybody|someone|somebody)|has\s+an\s+(?:assignee|owner))\b", text, re.I):
        return {"assignee_condition": "assigned"}
    if re.search(r"\b(?:my|mine|myself|assigned\s+to\s+(?:me|myself)|on\s+my\s+list)\b", text, re.I):
        return {"assignee_self": True, "assignee_condition": "self"}
    if re.search(r"\b(?:their|these users?|those users?|both users?)\b", text, re.I):
        return {"assignee_reference": "context"}
    possessive = re.search(
        rf"\b({_PERSON_TOKEN}(?:\s*(?:,|\band\b|&)\s*{_PERSON_TOKEN})*)['’]s\s+"
        r"(?:(?:(?:first|last)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)|"
        r"pending|open|outstanding|unfinished|completed|done|overdue|due|today|this\s+week|"
        r"high|low|urgent|priority|p[1-4]|first|second|third|fourth|fifth|last|latest|earliest)\s+){0,5}"
        r"(?:(?:action|todo|to-do)\s+)?(?:tasks?|items?|work)\b",
        text, re.I,
    )
    relation = re.search(
        rf"\b(?:assigned\s+to|belonging\s+to|owned\s+by|tasks?\s+(?:of|for)|items?\s+(?:of|for)|work\s+(?:of|for))\s+"
        rf"({_PERSON_TOKEN}(?:\s*(?:,|\band\b|&)\s*{_PERSON_TOKEN})*)",
        text, re.I,
    ) or re.search(
        rf"\b(?:tasks?|items?|ones?|work)\s+(?:of|from|for)\s+"
        rf"({_PERSON_TOKEN}(?:\s*(?:,|\band\b|&)\s*{_PERSON_TOKEN})*)",
        text, re.I,
    )
    tentative = False
    if not relation:
        relation = re.search(
            rf"\b(?:first|second|third|fourth|fifth|last|\d+(?:st|nd|rd|th))\s+"
            rf"({_PERSON_TOKEN})\s+(?:tasks?|items?|ones?)\b",
            text, re.I,
        )
        tentative = bool(relation)
    # A trailing Slack member reference after of/from is an unambiguous source
    # filter even when intervening state words occur ("first one as done of @X").
    if not relation and extract_contextual_reference(text):
        relation = re.search(
            rf"\b(?:of|from)\s+(<@[A-Z0-9]+(?:\|[^>]+)?>|@[A-Za-z][\w.-]*)\s*"
            rf"(?:\s+to\s+{_PERSON_TOKEN})?$",
            text, re.I,
        )
    match = possessive or relation
    if not match:
        return {}
    people, reference = _person_list(match.group(1))
    return {"assignees": people,
            **({"assignee_reference": reference} if reference else {}),
            **({"assignee_tentative": True} if tentative else {})}


def _filtered_selection_reference(text, filters):
    """Extract a selection whose neighboring member phrase defines candidates."""
    if not filters or not filters.get("assignee_tentative"):
        return None
    value = str(text or "")
    selection = (
        r"(?:first|last)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)|"
        r"first|second|third|fourth|fifth|last|\d+(?:st|nd|rd|th)"
    )
    attributive = re.search(
        rf"\b({selection})\s+{_PERSON_TOKEN}\s+(?:tasks?|items?|ones?)\b",
        value, re.I)
    if attributive:
        return parse_reference(attributive.group(1))
    return None


def _semantic_relationship_parse(text):
    """Parse reusable action/target/recipient grammar before narrow local parsers."""
    t = text.strip().rstrip(".?!")

    def assignment_metadata(value):
        """Remove field clauses and return independent mutation changes."""
        cleaned = value.strip()
        changes = []
        priority = _PRIORITY.search(cleaned)
        if priority:
            changes.append({"field": "priority", "value": priority.group(1).upper()})
            cleaned = cleaned[:priority.start()] + cleaned[priority.end():]
        else:
            natural_priority = re.search(
                r"\b(?:(?:make\s+it|set(?:\s+it)?(?:\s+to)?|with)?\s*)"
                r"(urgent|critical|highest|high|medium|normal|low|lowest)(?:\s+priority)?\b",
                cleaned, re.I,
            )
            if natural_priority:
                from config import normalize_priority
                normalized = normalize_priority(natural_priority.group(1))
                if normalized:
                    changes.append({"field": "priority", "value": normalized})
                    cleaned = cleaned[:natural_priority.start()] + cleaned[natural_priority.end():]
        due = re.search(
            r"\b(?:due(?:\s+date)?|by|deadline(?:\s+to)?)\s+"
            r"(.+?)(?=\s+(?:p[1-4]|urgent|critical|highest|high|medium|normal|low|lowest)(?:\s+priority)?\b|[,;]|$)",
            cleaned, re.I,
        )
        if due:
            normalized = _resolve_natural_date(due.group(1))
            if normalized:
                changes.append({"field": "due_date", "value": normalized})
                cleaned = cleaned[:due.start()] + cleaned[due.end():]
        cleaned = re.sub(r"\s*,\s*", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
        return cleaned, changes

    # Field continuations belong to the same assignment operation.
    assignment_text = t
    continuation_changes = []
    continuations = (
        r"\s+and\s+(?:push|move|change|set|update)\s+(?:the\s+)?(?:deadline|due\s+date)\s+to\s+(.+)$",
        r"\s+and\s+(?:make|set)\s+it\s+(p[1-4]|urgent|critical|highest|high|medium|normal|low|lowest)(?:\s+priority)?$",
    )
    for index, pattern in enumerate(continuations):
        continuation = re.search(pattern, assignment_text, re.I)
        if not continuation:
            continue
        if index == 0:
            value = _resolve_natural_date(continuation.group(1))
            if value:
                continuation_changes.append({"field": "due_date", "value": value})
        else:
            from config import normalize_priority
            value = normalize_priority(continuation.group(1).replace(" priority", ""))
            if value:
                continuation_changes.append({"field": "priority", "value": value})
        assignment_text = assignment_text[:continuation.start()].strip()
        break

    # Assignment is three independent dimensions: candidate filters, a
    # selection over those candidates, and the destination member. Split at
    # the final "to" so a source clause such as "assigned to Sam" remains in
    # the target rather than being consumed as part of the destination.
    # Destination-first syntax is structurally equivalent to the ordinary
    # target-to-destination form. Bare names require the determiner so an
    # arbitrary task title is never mistaken for a member.
    fronted = re.fullmatch(
        rf"(?:assign|give|hand|allocate)\s+"
        rf"(?:(me|myself|<@[A-Z0-9]+(?:\|[^>]+)?>|@[A-Za-z][\w.-]*)|({_PERSON_TOKEN}))\s+the\s+"
        rf"(.+)", assignment_text, re.I)
    assignment = None if fronted else re.fullmatch(r"(assign|reassign|transfer|move|hand|give|allocate)\s+(.+)\s+to\s+(.+)", assignment_text, re.I)
    dative = None if assignment or fronted else re.fullmatch(
        r"(?:give|hand)\s+(me|myself|<@[A-Z0-9]+(?:\|[^>]+)?>|@[A-Za-z][\w.-]*|[A-Za-z][\w.-]*)\s+(?:the\s+)?(.+)",
        assignment_text, re.I)
    if assignment or dative or fronted:
        if fronted:
            target, raw_people = fronted.group(3), fronted.group(1) or fronted.group(2)
        else:
            target, raw_people = ((assignment.group(2), assignment.group(3)) if assignment
                                  else (dative.group(2), dative.group(1)))
        # "give it back" carries the same contextual target as "give it".
        target = re.sub(r"\s+back$", "", target.strip(), flags=re.I)
        source_filter = {}
        source = re.fullmatch(rf"(.+?)\s+from\s+({_PERSON_TOKEN})", target, re.I)
        if source:
            target = source.group(1).strip()
            source_people, _ = _person_list(source.group(2))
            if source_people:
                source_filter = {"assignees": source_people}
        # Explicit creation language remains handled by the creation parser.
        if not re.search(r"\b(?:new\s+task|task\s+(?:called|named))\b", target, re.I):
            raw_people, inline_changes = assignment_metadata(raw_people)
            people, member_reference = _person_list(raw_people)
            if people and not member_reference:
                value = people if len(people) > 1 else people[0]
                target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.I).strip()
                changes = [{"field": "assignee", "value": value}, *inline_changes, *continuation_changes]
                result = _target_result("update", target, changes)
                filters = source_filter or _assignee_filter(target)
                if filters:
                    result.update(filters)
                    if filters.get("assignee_tentative"):
                        result["fallback_task_name"] = _clean_task(target)
                selection = extract_contextual_reference(target) or parse_reference(target)
                if not selection:
                    selection = _filtered_selection_reference(target, filters)
                explicit_collection = bool(re.search(r"\b(?:all|every|each|everything)\b", target, re.I))
                if selection:
                    result.update(task_name="__LAST__", reference=asdict(selection))
                    result["target_scope"] = "filtered" if filters else "contextual"
                    result["reference_scope"] = "filtered" if filters else "contextual"
                elif explicit_collection:
                    result.update(
                        task_name="__LAST__", reference=asdict(Reference("all")),
                        target_scope="filtered" if filters else "all_applicable",
                        reference_scope="filtered",
                    )
                if re.search(r"\b(?:pending|open|outstanding|unfinished)\b", target, re.I):
                    result.update(status="open", completed=False)
                elif re.search(r"\b(?:completed|done|finished)\b", target, re.I):
                    result.update(status="completed", completed=True)
                priority = _PRIORITY.search(target)
                if priority:
                    result["priority"] = priority.group(1).upper()
                if re.search(r"\boverdue\b", target, re.I):
                    result["overdue"] = True
                if re.search(r"\bdue\s+today\b", target, re.I):
                    result["due_today"] = True
                return result

    # Relationship syntax with the destination before the target.
    patterns = (
        rf"make\s+(.+?)\s+responsible\s+for\s+(.+)",
        rf"({_PERSON_TOKEN}(?:\s*(?:,|\band\b|&)\s*{_PERSON_TOKEN})*)\s+should\s+(?:handle|own|take)\s+(.+)",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, t, re.I)
        if not match:
            continue
        raw_people, target = match.group(1), match.group(2)
        people, reference = _person_list(raw_people)
        if not people and not reference:
            return {}
        result = _target_result("update", target, [{"field": "assignee", "value": people if len(people) > 1 else people[0]}])
        if reference:
            result["needs_clarification"] = True
            result["clarification"] = "Which Slack users should receive the task?"
        return result

    # Bulk operations combine one action, a set reference, and assignee filters.
    bulk = re.match(r"^(delete|remove|clear|complete|finish|close|reopen)\b\s+(.+)$", t, re.I)
    if bulk:
        target = bulk.group(2)
        filters = _assignee_filter(target)
        if filters and re.search(r"\b(?:all|every|everything|their|these|those)\b", target, re.I):
            action = bulk.group(1).casefold()
            intent = "delete" if action in {"delete", "remove", "clear"} else ("reopen" if action == "reopen" else "complete")
            result = _target_result(intent, "all")
            result.update(filters, reference_scope="filtered")
            if re.search(r"\b(?:pending|open|outstanding|unfinished)\b", target, re.I):
                result["completed"] = False
            elif re.search(r"\b(?:completed|done|finished)\b", target, re.I):
                result["completed"] = True
            return result

    # Read requests share the same assignee-filter extraction.
    if (re.match(r"^(?:show|list|find|search|display|give me|what(?:'s| is| are)|which)\b", t, re.I)
            and re.search(r"\b(?:tasks?|items?|action items?|todo|work|pending|open|outstanding|completed|done|overdue|due)\b", t, re.I)):
        targeted = _parse_information(t)
        if targeted and (targeted.get("reference") or {}).get("kind") != "all":
            return targeted
        filters = _assignee_filter(t)
        if filters:
            result = {"intent": "list", **filters}
            result["completed"] = True if re.search(r"\b(?:completed|done|finished)\b", t, re.I) else False
            if re.search(r"\b(?:all|everything|everyone(?:'s)?)\b", t, re.I) and not re.search(r"\b(?:pending|open|outstanding|completed|done|finished)\b", t, re.I):
                result["completed"] = None
                result["all_tasks"] = True
            if re.search(r"\boverdue\b", t, re.I):
                result["overdue"] = True
            if re.search(r"\bdue\s+today\b", t, re.I):
                result["due_today"] = True
            priority = _priority_filter(t)
            if priority:
                result["priority"] = priority
            return result
    return {}


def _parse_information(text):
    """Interrogative syntax separates targeted reads from collection queries."""
    t = text.strip().rstrip(".?!")
    target = None
    patterns = (
        r"(?:what(?:'?s| is)|show(?: me)?|tell me)\s+(?:the\s+)?(?:status|priority|assignee|due date|deadline)\s+(?:of|for)\s+(.+)",
        r"what(?:'?s| is)\s+(.+?)(?:'s|\s+)(?:status|priority|assignee|due date|deadline)",
        r"who\s+(?:is|was)\s+(?:assigned to|responsible for|working on)\s+(.+)",
        r"when\s+is\s+(.+?)\s+due",
        r"(?:did|have)\s+i\s+(?:finish|finished|complete|completed)\s+(.+)",
        r"is\s+(.+?)\s+(?:done|finished|completed|pending|open)",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, t, re.I)
        if match:
            target = match[1]
            break
    if target:
        return _target_result("inspect", target)
    # A bare ordinal/pronoun requests information unless answering clarification.
    if parse_reference(t):
        return _target_result("inspect", t)
    return {}


def _requested_statuses(text):
    """Return the status dimension without collapsing coordinated values."""
    values = []
    if _STATUS_OPEN.search(text):
        values.append("open")
    if _STATUS_COMPLETED.search(text):
        values.append("completed")
    return values


def _parse_progress_request(text):
    """Map progress concepts to metrics; calculation remains deterministic."""
    t = text.strip().rstrip(".?!")
    lower = t.casefold()
    if re.match(r"^(?:add|create|assign|reassign|move|update|change|edit|complete|finish|reopen|delete|remove)\b", lower):
        return {}
    # Analytics is a question/reporting speech act.  This boundary prevents
    # words such as "completion", "dashboard", or "finish" inside a task
    # title from reclassifying a create or mutation request as analytics.
    if not re.match(
            r"^(?:how|what|which|show|display|give|provide|summari[sz]e|report|compare|are|is|do|tell)\b",
            lower):
        return {}
    progress_signal = bool(re.search(
        r"\b(?:progress|insights?|analytics|dashboard|workload|completion\s+rate|pending\s+rate|"
        r"project\s+summary|work\s+summary|at\s+risk|how\s+(?:am|is|are)\b.+\bdoing|"
        r"how\s+much\s+work|how\s+many\b.+\b(?:complete|completed|pending))\b",
        lower))
    breakdown_signal = bool(
        re.search(r"\b(?:breakdown|distribution)\b", lower)
        and re.search(r"\b(?:status|priority|assignee|owner|workload)\b", lower)
    )
    timestamp_question = bool(re.search(
        r"\bwhat\s+did\s+(?:we|my\s+team|the\s+team)\b.+\b(?:complete|finish|create|add)\w*\b.+"
        r"\b(?:today|this\s+week|last\s+week|this\s+month|over\s+time)\b", lower))
    overdue_question = bool(re.search(r"\bare\s+there\b.+\boverdue\b", lower))
    status_values = _requested_statuses(lower)
    status_comparison = len(status_values) > 1 and bool(re.search(
        r"\b(?:vs\.?|versus|compare|comparison|difference|counts?|breakdown|distribution)\b", lower))
    status_summary = bool(re.search(
        r"\b(?:completion|task)\s+status\b|\bstatus\s+(?:summary|breakdown|distribution)\b", lower))
    comparison_question = bool(re.search(
        r"\bcompare\b.+\b(?:week|month|period)\b.+\b(?:week|month|period)\b", lower))
    if not (progress_signal or breakdown_signal or timestamp_question or overdue_question
            or comparison_question or status_comparison or status_summary):
        return {}

    metrics = []
    if re.search(r"\b(?:summary|progress|how\s+(?:am|is|are)\b.+\bdoing)\b", lower):
        metrics.append("summary" if "summary" in lower else "overview")
    if re.search(r"\b(?:completion\s+rate|pending\s+rate|how\s+many\b.+\b(?:complete|completed|pending))\b", lower):
        metrics.append("completion")
    if re.search(r"\b(?:workload|how\s+much\s+work)\b", lower):
        metrics.append("workload")
    if (status_comparison or status_summary or re.search(
            r"\bstatus\b.+\b(?:distribution|breakdown|summary)\b|\b(?:distribution|breakdown)\b.+\bstatus\b", lower)):
        metrics.append("status_distribution")
    if re.search(r"\bpriority\b.+\b(?:distribution|breakdown|summary)\b|\b(?:distribution|breakdown)\b.+\bpriority\b", lower):
        metrics.append("priority_distribution")
    if re.search(r"\boverdue\b", lower):
        metrics.append("overdue")
    if re.search(r"\bdue\s+today\b", lower):
        metrics.append("due_today")
    if re.search(r"\bdue\s+(?:this|in\s+the)\s+week\b", lower):
        metrics.append("due_this_week")
    if re.search(r"\b(?:upcoming|next)\b.+\b(?:deadline|due)\b", lower):
        metrics.append("upcoming")
    if re.search(r"\b(?:at\s+risk|risk(?:y|s)?)\b", lower):
        metrics.append("at_risk")
    if re.search(r"\b(?:completed?|finished)\b.+\b(?:today|this\s+week|last\s+week|this\s+month|over\s+time)\b", lower):
        metrics.append("completed_over_time")
    if re.search(r"\b(?:create(?:d)?|add(?:ed)?)\b.+\b(?:today|this\s+week|last\s+week|this\s+month|over\s+time)\b", lower):
        metrics.append("created_over_time")
    if comparison_question:
        metrics.append("comparison")
    if not metrics:
        metrics.append("overview")

    result = {"intent": "progress", "analytics_metrics": list(dict.fromkeys(metrics))}
    if status_values:
        result["statuses"] = status_values
    if status_comparison:
        result["result_operation"] = "comparison"
    priority = _priority_filter(t)
    if priority:
        result["priority"] = priority
    filters = _assignee_filter(t)
    if filters:
        result.update(filters)
    elif re.search(r"\b(?:my|mine|myself|how\s+am\s+i)\b", lower):
        result.update(assignee_self=True, assignee_condition="self")
    else:
        named = re.search(
            rf"\bhow\s+is\s+({_PERSON_TOKEN})\s+doing\b|"
            rf"\b({_PERSON_TOKEN})['’]s\s+(?:progress|workload|performance)\b|"
            rf"\b(?:progress|workload|performance)\s+(?:of|for)\s+({_PERSON_TOKEN})\b",
            t, re.I)
        team_scope = re.search(r"\b(?:we|our|ours|team|everyone|everybody)\b", lower)
        period_scope = re.search(r"\b(?:this|last|current|previous)\s+(?:week|month|quarter|year)\b", lower)
        if named and not team_scope and not period_scope:
            result["assignees"] = [named.group(1) or named.group(2) or named.group(3)]

    today = _today_date()
    if re.search(r"\bthis\s+week\b", lower):
        start = today - timedelta(days=today.weekday())
        result["analytics_period"] = {"start": start.isoformat(), "end": (start + timedelta(days=6)).isoformat()}
    elif re.search(r"\blast\s+week\b", lower):
        end = today - timedelta(days=today.weekday() + 1)
        result["analytics_period"] = {"start": (end - timedelta(days=6)).isoformat(), "end": end.isoformat()}
    elif re.search(r"\btoday\b", lower):
        result["analytics_period"] = {"start": today.isoformat(), "end": today.isoformat()}
    if result.get("analytics_period") and re.search(r"\b(?:progress|summary)\b", lower):
        result["analytics_metrics"] = list(dict.fromkeys(
            [*result["analytics_metrics"], "completed_over_time"]))
    if comparison_question:
        current_start = today - timedelta(days=today.weekday())
        previous_start = current_start - timedelta(days=7)
        result["analytics_comparison"] = {
            "current": {"start": current_start.isoformat(),
                        "end": (current_start + timedelta(days=6)).isoformat()},
            "previous": {"start": previous_start.isoformat(),
                         "end": (previous_start + timedelta(days=6)).isoformat()},
        }
    return result


def _parse_project_intelligence(text):
    """Normalize project-management concepts without making data decisions."""
    t = text.strip().rstrip(".?!")
    lower = t.casefold()
    if re.match(r"^(?:add|create|assign|reassign|move|update|change|edit|complete|finish|reopen|delete|remove)\b", lower):
        return {}
    if re.fullmatch(r"(?:yes[, ]*)?(?:confirm|confirmed|proceed|go\s+ahead)", lower):
        return {"intent": "confirm"}
    if re.fullmatch(r"(?:cancel|never\s+mind|nevermind|do\s+not\s+proceed|stop)", lower):
        return {"intent": "cancel"}
    if re.search(r"\b(?:apply|accept|use|execute)\b", lower) and re.search(
            r"\b(?:plan|proposal|suggestion|recommended|rebalanc(?:e|ing)|changes)\b", lower):
        return {"intent": "apply_proposal"}

    result = None
    if re.search(r"\bstand[- ]?up\b", lower):
        result = {"intent": "standup"}
    elif (re.search(r"\b(?:plan|schedule|organize|organise|prioritize|prioritise)\b", lower)
          and re.search(r"\b(?:tasks?|items?|work|week|days?|monday|friday)\b", lower)):
        result = {"intent": "plan"}
    elif (re.search(r"\b(?:overloaded|underloaded|overworked)\b", lower)
          or (re.search(r"\b(?:capacity|balance|rebalance|distribution)\b", lower)
              and re.search(r"\b(?:workload|tasks?|team|assignees?|members?|work)\b", lower))):
        result = {"intent": "workload", "recommend_balance": bool(re.search(
            r"\b(?:suggest|recommend|better|balance|rebalance|redistribute)\b", lower))}
    elif re.search(r"\b(?:task\s+health|health\s+of|need(?:s|ing)?\s+attention|on\s+track|health\s+status)\b", lower):
        result = {"intent": "health", "attention_only": bool(re.search(
            r"\b(?:need(?:s|ing)?\s+attention|at\s+risk|problematic|overdue)\b", lower))}
    elif (re.search(r"\b(?:who|what)\s+(?:changed|updated|modified)\b", lower)
          or re.search(r"\b(?:change|audit|mutation)\s+history\b", lower)):
        result = {"intent": "history"}
    elif re.search(r"\b(?:blocked|blockers?|depends?\s+on|dependencies|dependency|becomes?\s+available)\b", lower):
        result = {"intent": "dependencies"}
    if result is None:
        return {}

    named_target = None
    if result["intent"] == "health":
        match = re.search(
            r"\bwhy\s+is\s+(.+?)\s+(?:marked\s+as\s+)?(?:need(?:s|ing)?\s+attention|on\s+track|overdue)\b",
            t, re.I)
        named_target = match.group(1) if match else None
    elif result["intent"] == "history":
        match = (re.search(r"\bwho\s+(?:changed|updated|modified)\s+(.+)$", t, re.I)
                 or re.search(r"\bwhat\s+(?:changed|was\s+updated|was\s+modified)\s+(?:on|for)\s+(.+)$", t, re.I))
        named_target = match.group(1) if match else None
    elif result["intent"] == "dependencies":
        match = re.search(r"\bwhat\s+does\s+(.+?)\s+depend\s+on\b", t, re.I)
        named_target = match.group(1) if match else None
        availability = re.search(
            r"\bwhat\s+becomes?\s+available\s+if\s+(.+?)\s+(?:is\s+)?(?:completed?|done|finished)\b",
            t, re.I)
        if availability:
            origin = re.sub(r"^(?:the\s+)?", "", availability.group(1).strip(), flags=re.I)
            result["dependency_origin"] = origin
    if named_target and not parse_reference(named_target):
        named_target = re.sub(r"^(?:the\s+)?", "", named_target.strip(), flags=re.I)
        if named_target and named_target.casefold() not in {"task", "item", "action item"}:
            result.update(task_name=named_target, target_scope="single")

    filters = _assignee_filter(t)
    if filters:
        result.update(filters)
    elif re.search(r"\b(?:my|mine|myself)\b", lower):
        result.update(assignee_self=True, assignee_condition="self")
    priority = _priority_filter(t)
    if priority:
        result["priority"] = priority
    if _STATUS_OPEN.search(t):
        result.update(status="open", completed=False)
    elif _STATUS_COMPLETED.search(t) and result["intent"] not in {"standup", "history", "dependencies"}:
        result.update(status="completed", completed=True)
    if _OVERDUE.search(t):
        result["overdue"] = True
    if _DUE_TODAY.search(t):
        result["due_today"] = True
    if _DUE_WEEK.search(t):
        result["due_this_week"] = True

    reference = extract_contextual_reference(t)
    if not reference:
        contextual = re.search(
            r"\b(it|that|this|that\s+task|this\s+task|those|these|them|those\s+tasks|these\s+tasks|"
            r"the\s+previous\s+one|the\s+last\s+one)\b", t, re.I)
        reference = parse_reference(contextual.group(1)) if contextual else None
    if not reference:
        target = re.search(
            r"\b(?:health|history|dependencies|blockers?)\s+(?:of|for)\s+(.+)$", t, re.I)
        reference = parse_reference(target.group(1)) if target else None
    if reference:
        result.update(task_name="__LAST__", reference=asdict(reference), target_scope="contextual")

    if result["intent"] == "plan":
        today = _today_date()
        if re.search(r"\bnext\s+week\b", lower):
            start = today + timedelta(days=(7 - today.weekday()))
            end = start + timedelta(days=4)
        elif re.search(r"\b(?:this\s+week|monday\s+(?:through|to|until)\s+friday)\b", lower):
            start = today - timedelta(days=today.weekday())
            if today.weekday() > 4:
                start += timedelta(days=7)
            end = start + timedelta(days=4)
        else:
            start = today if today.weekday() < 5 else today + timedelta(days=7 - today.weekday())
            end = start + timedelta(days=6)
        result["planning_period"] = {"start": start.isoformat(), "end": end.isoformat()}
    return result


def _parse_field_change(text):
    """Parse field/target/value structure, not a list of complete example sentences."""
    t = text.strip().rstrip(".?!")
    match = re.fullmatch(r"(?:assign|reassign)\s+(.+?)\s+to\s+(.+)", t, re.I)
    if match and (parse_reference(match[1]) or t.lower().startswith("reassign")):
        people, reference = _person_list(match[2])
        if reference:
            return {"intent": "clarify", "clarification": "Which Slack users should receive the task?"}
        if people:
            value = people if len(people) > 1 else people[0]
            return _target_result("update", match[1], [{"field": "assignee", "value": value}])
    if not re.match(r"(?:change|set|update|move|reschedule|modify|edit|adjust)\b", t, re.I):
        return {}
    body = re.sub(r"^\w+\s+", "", t)
    field_pattern = r"(priority|due\s+date|deadline|status|assignee|owner|name|title)"
    target, raw_field, raw_value = None, None, None
    match = re.fullmatch(r"(?:the\s+)?" + field_pattern + r"(?:\s+of\s+(.+?))?\s+to\s+(.+)", body, re.I)
    if match:
        raw_field, target, raw_value = match[1], match[2] or "it", match[3]
    else:
        match = re.fullmatch(r"(.+?)(?:'s)?\s+" + field_pattern + r"\s+to\s+(.+)", body, re.I)
        if match:
            target, raw_field, raw_value = match[1], match[2], match[3]
        else:
            match = re.fullmatch(r"(.+?)\s+to\s+(p[1-4])", body, re.I)
            if match:
                target, raw_field, raw_value = match[1], "priority", match[2]
    if target:
        field = _FIELD_MAP.get(raw_field.lower())
        value = (_resolve_natural_date(raw_value) or raw_value) if field == "due_date" else raw_value
        return _target_result("update", target, [{"field": field, "value": value}])
    return {}


def _parse_intent_raw(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^i(?:\'d| would)\s+like\s+(?:you\s+)?to\s+", "", text, flags=re.I)
    text = re.sub(r"^(?:actually[, ]+)?(?:(?:can|could|would|will) you\s+)?(?:please\s+)?", "", text, flags=re.I)
    text = re.sub(r"[, ]+please[.!?]*$", "", text, flags=re.I)
    text = re.sub(r"^\s*/add\b\s*", "", text, flags=re.I).strip()

    if not text:
        return _empty_result()

    # A transfer can be phrased as a selection followed by an anaphoric
    # assignment. It remains one operation because the first clause only
    # identifies the target; it does not request an independent mutation.
    transfer = re.fullmatch(
        r"(?:take|use|select)\s+(.+?)\s+and\s+"
        r"(?:assign|give|move|transfer)\s+(?:it|that|this)(?:\s+(?:task|item))?\s+to\s+(.+)",
        text, re.I)
    if transfer:
        semantic = _semantic_relationship_parse(f"assign {transfer.group(1)} to {transfer.group(2)}")
        if semantic:
            result = _empty_result(text)
            result.update(semantic)
            return result

    # Distinct action clauses need the semantic parser to preserve their order.
    # This detects grammar boundaries and action classes, not complete phrases.
    action_word = r"(?:add|create|show|list|find|search|inspect|check|update|change|edit|assign|reassign|complete|finish|reopen|delete|remove|move|give|transfer)"
    compound_connector = r"(?:;|\n|\bthen\b|\band\s+(?:(?:also|then)\s+)?|\b(?:also|additionally|after\s+that)\b\s*)"
    compound = bool(re.search(rf"{compound_connector}\s*(?:please\s+)?{action_word}\b", text, re.I))
    member_domain = bool(re.search(r"\b(?:workspace\s+)?(?:member|members|role|roles|admin|admins|manager|managers|viewer|viewers)\b", text, re.I))

    if compound:
        boundary = rf"(?:;|\n|\bthen\b|\band\s+(?=(?:(?:also|then)\s+)?(?:please\s+)?{action_word}\b)|\b(?:also|additionally|after\s+that)\b(?=\s+(?:please\s+)?{action_word}\b))"
        clauses = [re.sub(r"^(?:also|then|please)\s+", "", part.strip(), flags=re.I)
                   for part in re.split(boundary, text, flags=re.I) if part.strip()]
        operations = [parse_intent(clause) for clause in clauses]
        if len(operations) >= 2 and all(operation.get("intent") not in {"compound", "out_of_scope", "temporarily_unavailable"}
                                        for operation in operations):
            return {"intent": "compound", "operations": operations, "raw_text": text}

    project_request = _parse_project_intelligence(text)
    if project_request:
        result = _empty_result(text)
        result.update(project_request)
        return result

    progress = _parse_progress_request(text)
    if progress:
        result = _empty_result(text)
        result.update(progress)
        return result

    semantic = None if compound or member_domain else _semantic_relationship_parse(text)
    if semantic:
        result = _empty_result(text)
        result.update(semantic)
        return result

    # Explicit mutation grammar has precedence over generic reference-aware
    # information parsing. Otherwise a collection reference such as an entire
    # task set can be reclassified as an inspection before reaching resolution.
    field_change = None if compound or member_domain else _parse_field_change(text)
    if field_change:
        result = _empty_result(text)
        result.update(field_change)
        return result

    # ── 1. Deterministic mutation parser (no Ollama needed) ──────────────────
    local_mut = None if compound or member_domain else _local_parse_mutation(text)
    if local_mut:
        result = _empty_result(text)
        result.update(local_mut)
        return result

    # Interrogatives describe reads even when they contain words such as
    # "assigned" or "finish" that also appear in mutation grammar.
    if not compound and not member_domain and re.match(r"^(?:what|who|when|which|how|why|did|have|is|are)\b", text, re.I):
        question = _parse_information(text)
        if question:
            result = _empty_result(text)
            result.update(question)
            return result

    # ── 2. Deterministic read parser ─────────────────────────────────────────
    local_read = None if compound or member_domain else _local_parse(text)
    if local_read:
        result = _empty_result(text)
        result.update(local_read)
        return result

    question = None if compound or member_domain else _parse_information(text)
    if question:
        result = _empty_result(text)
        result.update(question)
        return result

    # ── 2b. Deterministic CREATE parser (multi-task support) ─────────────────
    local_create = None if compound or member_domain else _local_parse_create(text)
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
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage
        options = {"model": OLLAMA_MODEL, "format": "json", "temperature": 0.0,
                   "client_kwargs": {"headers": {"Authorization": "Bearer " + OLLAMA_API_KEY}, "timeout": 30}}
        if os.getenv("OLLAMA_HOST"):
            options["base_url"] = os.environ["OLLAMA_HOST"]
        llm = ChatOllama(**options)
    except ImportError:
        logger.error("Install langchain-ollama to enable the model fallback")
        return {"intent": "temporarily_unavailable"}

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
            if not isinstance(parsed, dict) or parsed.get("intent") not in {
                "create", "list", "inspect", "progress", "health", "plan", "workload", "standup",
                "apply_proposal", "confirm", "cancel", "history", "dependencies",
                "update", "complete", "reopen", "delete", "members", "compound", "clarify", "out_of_scope"
            }:
                return {"intent": "clarify", "clarification": "Please clarify the task and action."}
            if re.match(r"^(?:what|who|when|which|how|why|did|have|is|are)\b", text, re.I) and parsed["intent"] in {"create", "update", "complete", "reopen", "delete"}:
                return {"intent": "clarify", "clarification": "Are you asking about a task, or requesting a change?"}
            parsed.pop("target_ids", None)
            result.update(parsed)
            if (not isinstance(result.get("tasks"), list) or not isinstance(result.get("changes"), list)
                    or not isinstance(result.get("operations"), list)):
                return {"intent": "clarify", "clarification": "Please clarify the task and action."}
            for task in result["tasks"]:
                if isinstance(task, dict):
                    task.pop("target_ids", None)
            # Normalize references from the target phrase, never title substrings.
            name = result.get("task_name")
            if name and name != "__LAST__" and not parse_reference(name):
                for key in ("selection", "selection_index", "selection_numbers", "selection_count", "task_reference", "reference"):
                    result.pop(key, None)
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
            logger.error("Ollama returned invalid JSON")
            return _empty_result(text)

        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                logger.warning("Ollama rate limit hit — not retrying")
                return {"intent": "temporarily_unavailable"}
            if attempt < 2:
                delay = 2 ** attempt
                logger.warning("Ollama transport error, retry %d in %ds", attempt + 1, delay)
                time.sleep(delay)
            else:
                logger.error("Ollama failed after 3 attempts")
                return {"intent": "temporarily_unavailable"}

    return {"intent": "temporarily_unavailable"}


def _normalize_target_structure(result: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Preserve collection meaning independently of whichever parser produced it."""
    if not isinstance(result, dict):
        return result
    scope = collection_scope(text)
    reference = result.get("reference") or {}
    semantic_filters = _assignee_filter(text)
    embedded_reference = extract_contextual_reference(text) or _filtered_selection_reference(text, semantic_filters)
    live_collection_query = result.get("intent") == "list" and scope in {"all_applicable", "filtered"}
    if embedded_reference and not live_collection_query and result.get("intent") in {
            "list", "inspect", "update", "complete", "reopen", "delete", "temporarily_unavailable"}:
        original_task_name = result.get("task_name")
        if result.get("intent") in {"list", "temporarily_unavailable"}:
            result["intent"] = "inspect"
        result["tasks"] = []
        result["task_name"] = "__LAST__"
        result["reference"] = asdict(embedded_reference)
        reference = result["reference"]
        # Member/status filters establish the candidate set; the reference is
        # then applied to that set. Preserve both dimensions independently.
        if semantic_filters and not (result.get("assignee") or result.get("assignees") or result.get("assignee_self")):
            result.update(semantic_filters)
        if semantic_filters.get("assignee_tentative") and not result.get("fallback_task_name"):
            result["fallback_task_name"] = original_task_name
        if semantic_filters and re.search(
                r"\b(?:one|ones|it|this|that|these|those|above|mentioned|discussed)\b", text, re.I):
            # Prefer the immutable displayed snapshot when the selection noun
            # itself is contextual. The resolver may only use live filtered
            # data when no displayed snapshot exists.
            result["candidate_source"] = "displayed"
    elif live_collection_query:
        # "all" in a live list query describes its scope. It is not a
        # positional reference to a previously displayed set.
        result["target_scope"] = scope
        result.pop("reference", None)
        result.pop("reference_scope", None)
        if result.get("task_name") == "__LAST__":
            result["task_name"] = None
    if reference and (result.get("assignee") or result.get("assignees")):
        result["target_scope"] = "filtered"
        result["reference_scope"] = "filtered"
    if scope and reference.get("kind") == "all":
        result["target_scope"] = scope
        if scope == "all_applicable":
            result["reference_scope"] = "filtered"
    # A creation-shaped parse with an assignee and a collection target is an
    # assignment mutation over existing items, never a new task title.
    if scope and result.get("intent") == "create" and result.get("assignee"):
        result["intent"] = "update"
        result["task_name"] = "__LAST__"
        result["reference"] = {"kind": "all", "positions": (), "count": 0}
        result["target_scope"] = scope
        result["reference_scope"] = "filtered" if scope == "all_applicable" else result.get("reference_scope")
        result["changes"] = [{"field": "assignee", "value": result["assignee"]}]
        if re.search(r"\b(?:pending|open|outstanding|unfinished)\b", text, re.I):
            result["completed"] = False
    return result


def _normalize_grouped_target_structure(result: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Represent coordinated task constraints as independent candidate groups."""
    if not isinstance(result, dict) or result.get("intent") not in {"update", "complete", "reopen", "delete"}:
        return result
    if result.get("target_groups") or result.get("task_name") is None:
        return result

    if result.get("intent") == "update":
        target_clause = str(result.get("task_name") or "")
    else:
        target_clause = re.sub(
            r"^(?:delete|remove|cancel|drop|erase|trash|complete|finish|close|reopen|re-open)\s+",
            "", text.strip().rstrip(".?!"), flags=re.I)
        filters = _assignee_filter(target_clause)
        possessive_owner = re.match(rf"^({_PERSON_TOKEN})['’]s\s+", target_clause, re.I)
        if not filters and possessive_owner:
            owners, _ = _person_list(possessive_owner.group(1))
            if owners:
                filters = {"assignees": owners}
        if filters and not (result.get("assignee") or result.get("assignees") or result.get("assignee_self")):
            result.update(filters)
        target_clause = re.sub(
            rf"\s+(?:assigned\s+to|from)\s+{_PERSON_TOKEN}\s*$", "", target_clause, flags=re.I)
        target_clause = re.sub(rf"^{_PERSON_TOKEN}['’]s\s+", "", target_clause, flags=re.I)
    target_clause = re.sub(r"\s+from\s*$", "", target_clause, flags=re.I).strip()

    parts = [part.strip() for part in re.split(r"\s*(?:,|\band\b|&)\s*", target_clause, flags=re.I)]
    explicit_nouns = all(re.search(r"\b(?:tasks?|items?|entries)\b", part, re.I) for part in parts)
    shared_plural = bool(re.search(r"\b(?:tasks|items|entries)\s*$", target_clause, re.I))
    if len(parts) < 2 or not (explicit_nouns or shared_plural):
        return result

    def group_selector(value):
        embedded = extract_contextual_reference(value)
        if embedded:
            return embedded
        selector = re.search(
            r"\b(first\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)|"
            r"last\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)|"
            r"first|second|third|fourth|fifth|last|both|all|every|each|\d+(?:st|nd|rd|th))\b",
            value, re.I)
        return parse_reference(selector.group(1)) if selector else None

    shared_reference = group_selector(target_clause)
    groups = []
    for part in parts:
        reference = group_selector(part) or shared_reference
        query = re.sub(
            r"\b(?:the|matching|first|second|third|fourth|fifth|last|both|all|every|each|"
            r"tasks?|items?|entries|ones?)\b", " ", part, flags=re.I)
        query = re.sub(r"\b\d+(?:st|nd|rd|th)?\b", " ", query, flags=re.I)
        query = re.sub(r"\s+", " ", query).strip(" ,")
        if not query:
            return result
        group = {"query": query}
        if reference:
            group["reference"] = asdict(reference)
        groups.append(group)
    result["target_groups"] = groups
    result["task_name"] = None
    result["target_scope"] = "multiple"
    result.pop("reference", None)
    result.pop("target_selection", None)
    result["result_operation"] = "select_many"
    return result


def _normalize_query_structure(result: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Preserve orthogonal read operations even when a parser handled filters first.

    This is a small semantic grammar for query operators, not a command-phrase
    table. The language model may emit the same fields for less regular wording.
    """
    if not isinstance(result, dict) or result.get("intent") in {
            "create", "progress", "health", "plan", "workload", "standup", "apply_proposal",
            "confirm", "cancel", "history", "dependencies",
            "update", "complete", "reopen", "delete", "compound", "members"}:
        return result
    t = text.strip().rstrip(".?!")
    task_domain = bool(re.search(r"\b(?:tasks?|items?|action\s+items?|todo|work|deadlines?|due\s+dates?)\b", t, re.I))
    read_form = bool(_READ_ANCHORS.match(t) or re.match(r"^(?:group|compare|count|rank|sort)\b", t, re.I))
    analytical = bool(re.search(
        r"\b(?:group(?:ed)?|breakdown|compare|count|how\s+many|number\s+of|rank|sort|"
        r"earliest|soonest|nearest|closest|latest|farthest|highest\s+priority|"
        r"lowest\s+priority|alphabetical|between|from\b.+\b(?:through|until|to))\b",
        t, re.I,
    ))
    if not (task_domain and read_form and analytical):
        return result

    # A query operator acts on a retrieved collection, even if an earlier
    # parser mistook a collection word for a contextual target.
    result["intent"] = "list"

    group = re.search(r"\b(?:group(?:ed)?|break(?:down)?)\s+(?:.+?\s+)?by\s+(assignee|owner|status|priority|due\s+date|deadline)\b", t, re.I)
    if group:
        result["group_by"] = {"owner": "assignee", "deadline": "due_date", "due date": "due_date"}.get(
            group.group(1).casefold(), group.group(1).casefold())

    if re.search(r"\b(?:how\s+many|number\s+of|count)\b", t, re.I):
        result["aggregate"] = "count"
    if re.match(r"^compare\b", t, re.I):
        result["aggregate"] = "count"
        result.setdefault("group_by", "assignee")

    sort_specs = (
        (r"\b(?:earliest|soonest|nearest|closest)\b", "due_date", "asc"),
        (r"\b(?:latest|farthest)\b", "due_date", "desc"),
        (r"\bhighest\s+priority\b", "priority", "asc"),
        (r"\blowest\s+priority\b", "priority", "desc"),
        (r"\b(?:alphabetical|by\s+name)\b", "name", "asc"),
    )
    for pattern, field, direction in sort_specs:
        if re.search(pattern, t, re.I):
            result.update(sort_by=field, sort_order=direction)
            break

    quantity = re.search(r"\b(?:top|first|show|give\s+me)?\s*(\d+)\b", t, re.I)
    if quantity:
        result["limit"] = int(quantity.group(1))
    else:
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        quantity = re.search(r"\b(?:top|first|show|give\s+me)?\s*(one|two|three|four|five|six|seven|eight|nine|ten)\b", t, re.I)
        if quantity and result.get("sort_by"):
            result["limit"] = words[quantity.group(1).casefold()]

    date_range = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)$", t, re.I) or re.search(
        r"\bfrom\s+(.+?)\s+(?:through|until|to)\s+(.+)$", t, re.I)
    if date_range:
        start = _resolve_natural_date(date_range.group(1))
        end = _resolve_natural_date(date_range.group(2))
        if start and end:
            result.update(date_from=start, date_to=end)

    filters = _assignee_filter(t)
    if filters and not (result.get("assignee") or result.get("assignees") or result.get("assignee_self")):
        result.update(filters)
    inverse = re.search(rf"\bdoes\s+({_PERSON_TOKEN})\s+have\b", t, re.I)
    if inverse and not (result.get("assignee") or result.get("assignees") or result.get("assignee_self")):
        result["assignees"] = [inverse.group(1)]

    priority = _PRIORITY.search(t)
    if priority:
        result["priority"] = priority.group(1).upper()
    if _OVERDUE.search(t):
        result["overdue"] = True
    elif _DUE_TODAY.search(t):
        result["due_today"] = True
    elif _DUE_WEEK.search(t):
        result["due_this_week"] = True
    if _STATUS_OPEN.search(t):
        result.update(status="open", completed=False)
    elif result.get("intent") != "complete" and _STATUS_COMPLETED.search(t):
        result.update(status="completed", completed=True)
    if _ALL_TASKS.search(t) and result.get("completed") is None:
        result.update(all_tasks=True, completed=None, target_scope="all_applicable")

    for key in ("date_from", "date_to"):
        value = result.get(key)
        if value and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
            normalized = _resolve_natural_date(str(value))
            if normalized:
                result[key] = normalized
    return result


def _normalize_selection_structure(result: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Normalize selection independently from candidate filters and output intent."""
    if not isinstance(result, dict) or result.get("intent") in {
            "create", "progress", "health", "plan", "workload", "standup", "apply_proposal",
            "confirm", "cancel", "history", "dependencies", "compound", "members", "out_of_scope", "clarify"}:
        return result
    existing = result.get("target_selection")
    if isinstance(existing, dict) and existing.get("mode"):
        return result

    reference = reference_from(result)
    if reference:
        if reference.kind in {"focus", "previous", "relative"} or (
                reference.kind == "positions" and len(reference.positions) == 1):
            mode, count = "one", 1
        elif reference.kind in {"positions", "both", "head", "tail"}:
            count = len(reference.positions) if reference.kind == "positions" else (reference.count or 2)
            mode = "one" if count == 1 else "many"
        else:
            mode, count = "collection", None
        result["target_selection"] = {"mode": mode, "order_by": "position", "count": count}
        return result

    t = text.strip().casefold().rstrip(".?!")
    semantic_filters = (_assignee_filter(text)
                        if result.get("intent") not in {"create", "inspect"} else {})
    if semantic_filters and not (result.get("assignee") or result.get("assignees") or result.get("assignee_self")):
        result.update(semantic_filters)
    if _STATUS_OPEN.search(t) and not result.get("statuses"):
        result.update(status="open", completed=False)
    elif (not result.get("statuses") and result.get("intent") != "complete"
          and _STATUS_COMPLETED.search(t)):
        result.update(status="completed", completed=True)
    has_task_noun = bool(re.search(r"\b(?:tasks?|items?|action\s+items?|entries)\b", t))
    has_filter = bool(result.get("assignee") or result.get("assignees") or result.get("assignee_self")
                      or result.get("status") or result.get("priority") or result.get("date_from")
                      or result.get("date_to") or result.get("due_today") or result.get("overdue")
                      or result.get("due_this_week"))
    direct_selector = re.search(
        r"\b(?P<selector>most\s+recent|recent|latest|newest|oldest|earliest|nearest|soonest|"
        r"highest|lowest)\s+(?:(?:assigned|assignee|owned|due|priority|open|pending)\s+){0,3}"
        r"(?:tasks?|items?|action\s+items?|entries)\b", t)
    loose_filtered_selector = has_filter and has_task_noun and re.search(
        r"\b(most\s+recent|recent|latest|newest|oldest|earliest|nearest|soonest|highest|lowest)\b", t)
    selector = (direct_selector.group("selector") if direct_selector else
                (loose_filtered_selector.group(1) if loose_filtered_selector else None))
    if selector:
        due_semantics = bool(re.search(r"\b(?:due|deadline)\b", t))
        priority_semantics = bool(re.search(r"\bpriority\b", t) or selector in {"highest", "lowest"})
        if priority_semantics:
            order_by, direction = "priority", "asc" if selector == "highest" else "desc"
        elif due_semantics or selector in {"nearest", "soonest"}:
            order_by = "due_date"
            direction = "desc" if selector == "latest" else "asc"
        else:
            order_by = "created_at"
            direction = "asc" if selector in {"oldest", "earliest"} else "desc"
        requested_count = result.get("limit") if isinstance(result.get("limit"), int) and result["limit"] > 0 else 1
        result["target_selection"] = {
            "mode": "one" if requested_count == 1 else "many", "order_by": order_by,
            "direction": direction, "count": requested_count,
        }
        result["task_name"] = None
        result.pop("literal_name", None)
        result["target_scope"] = "filtered" if has_filter else "contextual"
        return result

    limit = result.get("limit")
    if isinstance(limit, int) and limit > 0:
        result["target_selection"] = {
            "mode": "one" if limit == 1 else "many",
            "order_by": result.get("sort_by") or "position",
            "direction": result.get("sort_order", "asc"),
            "count": limit,
        }
    elif result.get("target_scope") in {"filtered", "all_applicable", "contextual"}:
        result["target_selection"] = {"mode": "collection"}
    return result


def _normalize_semantic_dimensions(result: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Separate temporal, qualitative, and actor concepts before execution."""
    if not isinstance(result, dict):
        return result
    statuses = _requested_statuses(text)
    if len(statuses) > 1:
        result["statuses"] = statuses
        result["status"] = None
        result["completed"] = None
        if result.get("intent") == "list":
            result["all_tasks"] = True
    result.setdefault("actor", "requester")
    if result.get("intent") in {
            "progress", "health", "plan", "workload", "standup", "apply_proposal",
            "confirm", "cancel", "history", "dependencies"}:
        # Progress periods apply to real creation/completion timestamps inside
        # the analytics engine. They must never become due-date task filters.
        return result
    t = text.strip().casefold().rstrip(".?!")
    if result.get("intent") in {"out_of_scope", "temporarily_unavailable"} and re.search(
            r"\b(?:tasks?|items?|action\s+items?|work)\b", t) and re.search(
            r"\b(?:due|deadline|created?|added?|updated?|changed?|completed?|pending|priority)\b", t):
        result["intent"] = "list"
    semantic_filters = (_assignee_filter(text)
                        if result.get("intent") not in {"create", "inspect"} else {})
    if semantic_filters and not (result.get("assignee") or result.get("assignees") or result.get("assignee_self")):
        result.update(semantic_filters)
    if _STATUS_OPEN.search(t) and not result.get("statuses"):
        result.update(status="open", completed=False)
    priority = _priority_filter(text)
    if result.get("intent") == "list" and priority and not result.get("priority"):
        result["priority"] = priority

    if result.get("intent") == "list" and not result.get("query"):
        related = re.search(
            r"\b(?:find|show|list|search(?:\s+for)?)\s+(?:all\s+)?(.+?)[- ]related\s+"
            r"(?:tasks?|items?|action\s+items?)\b", text, re.I)
        if related:
            query = re.sub(
                r"\b(?:all|pending|open|incomplete|unfinished|completed|done|overdue|"
                r"p[1-4]|high[- ]priority|medium[- ]priority|low[- ]priority)\b",
                " ", related.group(1), flags=re.I)
            result["query"] = re.sub(r"\s+", " ", query).strip(" ,.-")

    # Normalize a temporal period separately from the field it constrains.
    if result.get("intent") == "list" and not result.get("temporal_filter"):
        field_match = re.search(r"\b(due|deadline|created?|added?|updated?|changed?|completed?|finished)\b", t)
        period_match = re.search(r"\b(today|tomorrow|this\s+week|last\s+week|this\s+month|last\s+month)\b", t)
        if field_match and period_match:
            word, period = field_match.group(1), period_match.group(1)
            if word in {"due", "deadline"}:
                temporal_field = "due_date"
            elif word.startswith(("creat", "add")):
                temporal_field = "created_at"
            elif word.startswith(("updat", "chang")):
                temporal_field = "updated_at"
            else:
                temporal_field = "completed_at"
            today = _today_date()
            if period == "today":
                temporal = {"field": temporal_field, "relation": "on", "date": today.isoformat()}
            elif period == "tomorrow":
                temporal = {"field": temporal_field, "relation": "on",
                            "date": (today + timedelta(days=1)).isoformat()}
            elif period in {"this week", "last week"}:
                start = today - timedelta(days=today.weekday())
                if period == "last week":
                    start -= timedelta(days=7)
                temporal = {"field": temporal_field, "relation": "between",
                            "date_from": start.isoformat(), "date_to": (start + timedelta(days=6)).isoformat()}
            else:
                anchor = today.replace(day=1)
                if period == "last month":
                    anchor = (anchor - timedelta(days=1)).replace(day=1)
                next_month = (anchor.replace(day=28) + timedelta(days=4)).replace(day=1)
                temporal = {"field": temporal_field, "relation": "between",
                            "date_from": anchor.isoformat(),
                            "date_to": (next_month - timedelta(days=1)).isoformat()}
            result["temporal_filter"] = temporal
            result["due_today"] = temporal_field == "due_date" and period == "today"
            result["due_this_week"] = temporal_field == "due_date" and period == "this week"

    # A date has meaning only together with the task field it constrains.
    if not result.get("temporal_filter") and re.search(r"\btoday\b", t):
        if re.search(r"\b(?:due|deadline)\b", t):
            temporal_field = "due_date"
        elif result.get("completed") is True or re.search(r"\b(?:finish(?:ed)?|completed?|done)\b", t):
            temporal_field = "completed_at"
        elif re.search(r"\b(?:created?|added?)\b", t):
            temporal_field = "created_at"
        elif re.search(r"\b(?:updated?|changed?|edited?)\b", t):
            temporal_field = "updated_at"
        else:
            temporal_field = "due_date"
        result["temporal_filter"] = {"field": temporal_field, "relation": "on", "date": _today()}
        result["due_today"] = temporal_field == "due_date"

    # Urgency is a ranking over available task attributes, not a collection
    # filter: higher priority wins, then the nearest due date.
    qualitative = re.search(r"\bmost\s+(?:urgent|important|critical)\b", t)
    if qualitative:
        if result.get("intent") in {"out_of_scope", "temporarily_unavailable"} and re.search(
                r"\b(?:tasks?|items?|action\s+items?)\b", t):
            result["intent"] = "list"
        result["target_selection"] = {
            "mode": "one", "order_by": "urgency", "direction": "asc", "count": 1}
        result["target_scope"] = "filtered" if (
            result.get("assignee") or result.get("assignees") or result.get("assignee_self")
            or result.get("status") or result.get("priority") or result.get("completed") is not None
        ) else "contextual"
        result["task_name"] = None
    return result


def _normalize_result_operation(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict) or result.get("result_operation"):
        return result
    if result.get("aggregate") or result.get("count_only"):
        operation = "comparison" if result.get("group_by") else "aggregate"
    else:
        mode = (result.get("target_selection") or {}).get("mode")
        operation = {"one": "select_one", "many": "select_many",
                     "collection": "return_collection"}.get(mode)
        if not operation:
            operation = "select_one" if result.get("intent") == "inspect" else "return_collection"
    result["result_operation"] = operation
    return result


def parse_intent(text: str) -> Dict[str, Any]:
    result = _normalize_query_structure(_parse_intent_raw(text), text)
    result = _normalize_target_structure(result, text)
    result = _normalize_grouped_target_structure(result, text)
    result = _normalize_semantic_dimensions(result, text)
    result = _normalize_selection_structure(result, text)
    return _normalize_result_operation(result)


# Backwards-compatible aliases
def parse(text: str) -> Dict[str, Any]:
    return parse_intent(text)

def parse_request(text: str) -> Dict[str, Any]:
    return parse_intent(text)

def parse_command(text: str) -> Dict[str, Any]:
    return parse_intent(text)
