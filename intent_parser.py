import os
import json
import logging
from typing import Any, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:cloud").strip()

def _empty_result(raw_text=""):
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

SYSTEM_PROMPT = """
You are a strict JSON intent parser for a Slack List bot managing action items.
You will receive a user's natural language request and must extract their intent and parameters into a valid JSON object.

Allowed intents: "create", "list", "update", "complete", "reopen", "delete", "out_of_scope", "clarify"

IMPORTANT RULES:
1. ONLY return valid JSON. Do not return markdown blocks or any other text.
2. DO NOT invent Slack User IDs, Task IDs, or Field IDs. Use exactly what the user says for names. If the user mentions <@U12345> or @U12345, output "U12345" as the assignee. The task_name MUST be the actual name of the work (e.g., "Client Report") and MUST NEVER be a Slack User ID like U12345 or <@U12345>.
3. If the user asks something completely unrelated to Slack Lists or tasks (e.g., "What is the weather?"), intent must be "out_of_scope".
4. If multiple matching items are possible but ambiguous (e.g., "delete the first 4 tasks"), you can return intent "list" or "delete" with "selection_count": 4.
5. Do NOT include words like "assign", "task", "for", "with" in the task_name. E.g. "Create a task called API7 TEST TASK for Aastha with priority P1" -> task_name: "API7 TEST TASK".
6. Priorities must be normalized to "P1", "P2", "P3", or "P4" if mentioned. Map natural expressions like "urgent" or "high priority" to "P1" or "P2", "medium priority" to "P3", and "low priority" to "P4". If it cannot be understood, leave it null.
7. Due dates should be formatted as YYYY-MM-DD if possible.
8. Status should be "open", "completed", or "in progress".
9. For "update", you must provide a "changes" array of dicts containing "field" and "value". e.g., [{"field": "assignee", "value": "Praveen"}, {"field": "priority", "value": "P1"}]
10. For pronouns like "this", "that", "the first one", set task_name to "__LAST__" and/or provide "selection": "first", "both", "all".
11. If the user refers to themselves ("my tasks", "assign to me"), set "assignee_self": true.
12. For "list" intent, populate "query" for search terms, "overdue" (bool), "due_today" (bool), "due_this_week" (bool), or "completed" (true/false) as appropriate.
13. If the user intent is ambiguous, return "clarify" and put the question in "clarification".
14. For general questions about tasks, assignments, deadlines, or status (e.g., "who is assigned to X?", "how many tasks do I have?", "what is the priority of Y?"), use the "list" intent with an appropriate query or assignee, so the bot can fetch the items and show them. Do NOT return out_of_scope. Handle ALL task-related queries by extracting the relevant parameters and mapping them to the closest intent (usually 'list' or 'clarify').

JSON SCHEMA:
{
  "intent": "create" | "list" | "update" | "complete" | "reopen" | "delete" | "out_of_scope" | "clarify",
  "task_name": "Clean Task Title",
  "priority": "P1",
  "status": "open",
  "assignee": "Aastha",
  "due_date": "2026-09-20",
  "completed": false,
  "selection": "single" | "first" | "all" | "both" | "numbered",
  "selection_count": 4,
  "changes": [{"field": "assignee", "value": "Praveen"}],
  "clarification": "Question string",
  "assignee_self": true,
  "query": "search query string",
  "overdue": false,
  "due_today": false,
  "due_this_week": false
}
"""

def parse_intent(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    # Strip Slack bot mentions
    import re
    text = re.sub(r"^\s*<@[A-Z0-9]+(?:\|[^>]+)?>\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*/add\b\s*", "", text, flags=re.I).strip()
    
    if not text:
        return _empty_result()

    def _local_parse(t: str) -> Dict[str, Any]:
        t_lower = t.lower().strip()
        
        # Don't intercept obvious mutations
        if any(w in t_lower for w in ["create", "add", "update", "set", "change", "delete", "remove"]):
            return {}

        res = {"intent": "list"}
        matched_filter = False

        # Priorities
        p_match = re.search(r"\b(p[1-4])\b", t_lower)
        if p_match:
            res["priority"] = p_match.group(1).upper()
            matched_filter = True

        # Overdue
        if "overdue" in t_lower:
            res["overdue"] = True
            matched_filter = True

        # Due today
        if re.search(r"\b(today|focus)\b", t_lower):
            res["due_today"] = True
            matched_filter = True

        # Status
        if "completed" in t_lower or "done" in t_lower:
            res["status"] = "completed"
            matched_filter = True
        elif "pending" in t_lower or "open" in t_lower:
            res["status"] = "open"
            matched_filter = True

        # Assignee
        assignee_match = re.search(r"<@([A-Z0-9]+)(?:\|[^>]+)?>", t)
        if assignee_match:
            res["assignee"] = f"<@{assignee_match.group(1)}>"
            matched_filter = True
        elif re.search(r"\b(my|i|me)\b", t_lower):
            res["assignee_self"] = True
            matched_filter = True
            
        # Is it a list query?
        is_list = bool(re.search(r"\b(list|show|get|what|which|who)\b", t_lower))
        is_generic_all = re.fullmatch(r"(list|show|get)?\s*(all)?\s*(tasks|task|action items?)", t_lower) or t_lower in ("list", "show", "all")

        # Return if it's explicitly generic OR if we matched specific filters
        if is_generic_all or (is_list and matched_filter) or (matched_filter and "task" in t_lower):
            if is_generic_all and not matched_filter:
                res["status"] = "open" # Default to pending for generic "list all"
            return res

        return {}

    local_intent = _local_parse(text)
    if local_intent:
        res = _empty_result(text)
        res.update(local_intent)
        return res

    if not OLLAMA_API_KEY:
        logger.error("OLLAMA_API_KEY is missing!")
        return _empty_result(text)

    llm = ChatOllama(model=OLLAMA_MODEL, format="json", temperature=0.0)

    import time
    for attempt in range(3):
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=text)
            ]
            response = llm.invoke(messages)
            output = response.content.strip()
            parsed = json.loads(output)
            parsed["raw_text"] = text
            return parsed
        except Exception as e:
            if attempt < 2:
                delay = 2 ** attempt
                logger.warning(f"Ollama API Error. Retrying in {delay}s... ({e})")
                time.sleep(delay)
                continue
            logger.exception(f"Failed to parse intent via Ollama for input: {text}")
            return {"intent": "temporarily_unavailable"}

def parse(text: str) -> Dict[str, Any]:
    return parse_intent(text)

def parse_request(text: str) -> Dict[str, Any]:
    return parse_intent(text)

def parse_command(text: str) -> Dict[str, Any]:
    return parse_intent(text)