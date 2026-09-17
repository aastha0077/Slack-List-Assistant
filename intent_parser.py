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

def _today() -> str:
    """Return today's date in Asia/Kathmandu as YYYY-MM-DD."""
    return datetime.now(ZoneInfo("Asia/Kathmandu")).date().isoformat()


def _tomorrow() -> str:
    d = datetime.now(ZoneInfo("Asia/Kathmandu")).date() + timedelta(days=1)
    return d.isoformat()


def _yesterday() -> str:
    d = datetime.now(ZoneInfo("Asia/Kathmandu")).date() - timedelta(days=1)
    return d.isoformat()


def _resolve_natural_date(text: str) -> Optional[str]:
    t = text.casefold()
    if re.search(r"\btoday\b", t): return _today()
    if re.search(r"\btomorrow\b", t): return _tomorrow()
    if re.search(r"\byesterday\b", t): return _yesterday()
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m: return m.group(1)
    months = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
              r"|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
    m = re.search(rf"\b({months})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", t, re.I) or \
        re.search(rf"\b(\d{{1,2}})\s+({months})\s+(\d{{4}})\b", t, re.I)
    if m:
        try:
            raw = m.group(0)
            for fmt in ("%B %d, %Y", "%B %d %Y", "%d %B %Y"):
                try:
                    return datetime.strptime(raw.title().replace(",", ""), fmt.replace(",", "")).date().isoformat()
                except ValueError:
                    pass
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Empty / error results
# ---------------------------------------------------------------------------

def _empty_result(raw_text: str = "") -> Dict[str, Any]:
    return {
        "intent": "out_of_scope",
        "task_name": None,
        "priority": None,
        "status": None,
        "assignee": None,
        "due_date": None,
        "completed": None,
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

JSON SCHEMA (omit null/false/missing fields):
{
  "intent": "create",
  "task_name": "...",
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
_COMPLETE_ANCHOR = re.compile(r"^(?:complete|finish|done|mark\s+(?:as\s+)?(?:done|complete|completed|finished)|close)\b", re.I)
_REOPEN_ANCHOR   = re.compile(r"^(?:reopen|re-open|uncheck|uncomplete|undo\s+complete)\b", re.I)

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
    name = name.strip(" .,;:\"'")
    return _TASK_NOISE.sub("", name).strip()


def _local_parse_mutation(text: str) -> Dict[str, Any]:
    """
    Deterministic parser for mutation intents (delete/update/complete/reopen).
    Returns populated dict or {} to fall through to Ollama.
    """
    t = text.strip()

    if _DELETE_ANCHOR.match(t):   verb = "delete"
    elif _UPDATE_ANCHOR.match(t): verb = "update"
    elif _COMPLETE_ANCHOR.match(t): verb = "complete"
    elif _REOPEN_ANCHOR.match(t): verb = "reopen"
    else: return {}

    # Strip the leading verb word(s)
    body = re.sub(r"^\S+\s*", "", t, count=1).strip()
    body = re.sub(r"^(?:as\s+)?(?:done|complete|completed|finished)\s*", "", body, flags=re.I).strip()

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
        r: Dict[str, Any] = {"intent": intent, "raw_text": text}
        if task_name:
            r["task_name"] = task_name
        if changes:
            r["changes"] = changes
        if assignee_raw:
            r["assignee"] = assignee_raw
        if assignee_self:
            r["assignee_self"] = True
        return r

    # ── complete / reopen — just need the task name ──
    if verb in ("complete", "reopen"):
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
# Deterministic local READ parser — regex patterns
# ---------------------------------------------------------------------------

# Mutations — never intercept in the read parser
_READ_ANCHORS = re.compile(
    r"^(list|show|get|display|fetch|what|which|who|how\s+many|tell\s+me|give\s+me|find)\b",
    re.I,
)
_WHAT_SHOULD = re.compile(
    r"^what\s+(should|can|must|do|does|will|would|am|are|is|have)\b",
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
    r"my\s+(tasks?|action\s+items?|work|todo|next|nearest|deadline)|"
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
_STATUS_COMPLETED = re.compile(r"\b(completed?|done|finished|closed)\b", re.I)
_STATUS_OPEN = re.compile(r"\b(pending|open|incomplete|not\s+done|not\s+completed?|in\s+progress)\b", re.I)

_DUE_TODAY = re.compile(
    r"\b(today|due\s+today|for\s+today|this\s+day)\b",
    re.I,
)
_OVERDUE = re.compile(r"\b(overdue|past\s+due|late|missed\s+deadline)\b", re.I)
_DUE_WEEK = re.compile(r"\b(this\s+week|this\s+week'?s?|week)\b", re.I)

_ALL_TASKS = re.compile(
    r"\b(all\s+(tasks?|action\s+items?|everything)|every\s+(task|action\s+item)|"
    r"show\s+everything|list\s+everything)\b",
    re.I,
)
_BARE_LIST = re.compile(r"^(list|show|get)(\s+all)?\s+(tasks?|action\s+items?)\s*$", re.I)

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
            or _COMPLETE_ANCHOR.match(t) or _REOPEN_ANCHOR.match(t)):
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
    elif _STATUS_OPEN.search(t):
        res["status"] = "open"
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
    is_generic = bool(_BARE_LIST.match(t.lower()) or _ALL_TASKS.search(t) or t.lower() in ("list", "show"))
    is_who = bool(_WHO_BARE.match(t))

    has_filter = any(k in res for k in (
        "due_today", "overdue", "due_this_week", "status", "priority",
        "assignee", "assignee_self", "sort_by", "limit", "completed",
    ))

    if is_generic or is_who or has_filter:
        if (is_generic or is_who) and len(res) == 1:
            res["status"] = "open"
        return res

    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_intent(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    # Strip leading Slack bot mention
    text = re.sub(r"^\s*<@[A-Z0-9]+(?:\|[^>]+)?>\s*", "", text, flags=re.I)
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
            parsed["raw_text"] = text
            if parsed.get("due_date") in ("today", "TODAY"):
                parsed["due_date"] = _today()
            elif parsed.get("due_date") in ("tomorrow", "TOMORROW"):
                parsed["due_date"] = _tomorrow()
            return parsed

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
