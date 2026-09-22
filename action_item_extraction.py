"""LLM interpretation plus deterministic normalization of transcript actions."""
from dataclasses import asdict, dataclass
from datetime import date
import logging
import re

import config
import intent_parser
from safe_diagnostics import log_exception, redact
from workflow_safety import title_similarity


logger = logging.getLogger(__name__)


class ExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractedAction:
    title: str
    assignee: str | None
    due_date: str | None
    priority: str | None
    status: str
    source_type: str
    source_reference: str
    confidence: float
    evidence: str
    clarification: str | None = None
    operation: str = "create"


SYSTEM_PROMPT = """You extract concrete future action items from meeting or transcript text.
Return one JSON object with key "items" containing a list. Each item must have exactly:
title, assignee, due_date, priority, status, confidence, evidence, clarification, operation.
- Use a concise imperative task title without assignee/date/priority words.
- assignee is a name or Slack mention explicitly supported by the transcript, otherwise null.
- due_date is YYYY-MM-DD only when explicitly supported; resolve relative dates against CURRENT_DATE.
- priority is P1/P2/P3/P4 only when explicitly stated, otherwise null.
- status is pending for future work. Do not turn already-completed work into a new task.
- operation is create unless the speaker explicitly asks to update, complete, or reopen an
  existing task. Never infer an update from a similar title alone.
- confidence is 0..1. Tentative suggestions and ambiguous pronouns must be below 0.75.
- evidence is a short verbatim supporting excerpt, at most 180 characters.
- clarification explains a real ambiguity; otherwise null.
Resolve pronouns only when the antecedent is unambiguous in the supplied context.
Merge repeated mentions of the same action. Never invent people, dates, priorities, or tasks.
If there are no concrete action items, return {"items": []}.
CURRENT_DATE: {current_date}
"""


def chunk_text(text, max_chars=12000, overlap_chars=600):
    text = str(text or "").strip()
    if not text:
        return []
    overlap_chars = min(max(0, overlap_chars), max_chars // 4)
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n", start + max_chars // 2, end),
                           text.rfind(" ", start + max_chars // 2, end))
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def _normalize(raw, source_type, source_reference, today):
    if not isinstance(raw, dict):
        raise ExtractionError("The extraction model returned an invalid action item.")
    title = str(raw.get("title") or "").strip()
    if not title or len(title) > 300:
        raise ExtractionError("An extracted action item has no valid title.")
    assignee = str(raw["assignee"]).strip() if raw.get("assignee") else None
    priority = config.normalize_priority(raw.get("priority")) if raw.get("priority") else None
    due = raw.get("due_date")
    if due:
        due = intent_parser._resolve_natural_date(str(due), today=today)
        if not due:
            raise ExtractionError(f"The extracted due date for {title!r} is invalid.")
    meta = {"assignee": assignee, "due_date": due, "priority": priority}
    title = intent_parser._clean_single_task_name(title, meta)
    if not title:
        raise ExtractionError("An extracted task title contained only control fields.")
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = re.sub(r"\s+", " ", str(raw.get("evidence") or "")).strip()[:240]
    clarification = str(raw.get("clarification") or "").strip() or None
    operation = str(raw.get("operation") or raw.get("intent") or "create").strip().casefold()
    if operation not in {"create", "update", "complete", "reopen"}:
        operation = "create"
    if operation == "update" and not any((assignee, due, priority)):
        clarification = clarification or f"The requested update for {title!r} does not specify a field value."
        confidence = min(confidence, 0.49)
    if due and due < today.isoformat():
        clarification = clarification or f"The stated due date {due} is in the past."
        confidence = min(confidence, 0.49)
    return ExtractedAction(title, assignee, due, priority, "pending", source_type,
                           source_reference, confidence, evidence, clarification, operation)


def _deduplicate(items):
    def comparable(title):
        return re.sub(r"\b(?:a|an|the)\b", " ", title, flags=re.I)
    merged = []
    for item in items:
        match_index = next((index for index, existing in enumerate(merged)
                            if title_similarity(comparable(item.title), comparable(existing.title)) >= 0.92
                            and item.operation == existing.operation
                            and (not item.assignee or not existing.assignee or item.assignee == existing.assignee)), None)
        if match_index is None:
            merged.append(item)
            continue
        existing = merged[match_index]
        conflicts = []
        for label, earlier, later in (
                ("assignee", existing.assignee, item.assignee),
                ("due date", existing.due_date, item.due_date),
                ("priority", existing.priority, item.priority)):
            if earlier and later and earlier != later:
                conflicts.append(label)
        evidence = existing.evidence
        if item.evidence and item.evidence not in evidence:
            evidence = (evidence + " / " + item.evidence).strip(" /")[:240]
        clarification = existing.clarification or item.clarification
        confidence = max(existing.confidence, item.confidence)
        if conflicts:
            clarification = clarification or (
                "Repeated transcript mentions conflict on " + ", ".join(conflicts) + ".")
            confidence = min(confidence, 0.49)
        merged[match_index] = ExtractedAction(
            existing.title, existing.assignee or item.assignee, existing.due_date or item.due_date,
            existing.priority or item.priority, existing.status, existing.source_type,
            existing.source_reference, confidence, evidence, clarification, existing.operation)
    return merged


def extract(contents, today=None, model=None):
    today = today or date.today()
    model = model or intent_parser.structured_model_json
    results = []
    for content_index, content in enumerate(contents, 1):
        chunks = chunk_text(content.text)
        for chunk_index, chunk in enumerate(chunks, 1):
            model_path = (getattr(model, "__module__", "") + "." +
                          getattr(model, "__qualname__", getattr(model, "__name__", type(model).__name__))).strip(".")
            logger.info(
                "Action-item extraction function=extract model_path=%s source_type=%s "
                "content_index=%d chunk=%d/%d input_chars=%d",
                model_path, content.source_type, content_index, chunk_index, len(chunks), len(chunk),
            )
            try:
                payload = model(SYSTEM_PROMPT.replace("{current_date}", today.isoformat()), chunk)
            except Exception as exc:
                log_exception(logger, "Action-item extraction failed", exc,
                              function="action_item_extraction.extract", model_path=model_path,
                              source_type=content.source_type, content_index=content_index,
                              chunk=f"{chunk_index}/{len(chunks)}")
                raise ExtractionError("The AI action-item extraction service failed.") from exc
            values = payload.get("items")
            if not isinstance(values, list):
                raise ExtractionError("The AI extraction result did not contain an action-item list.")
            for raw in values:
                # Completed historical statements are context, not new work.
                if (isinstance(raw, dict)
                        and config.normalize_status(raw.get("status")) == "completed"
                        and str(raw.get("operation") or raw.get("intent") or "").casefold()
                        not in {"complete", "reopen"}):
                    continue
                item = _normalize(raw, content.source_type, content.source_reference, today)
                logger.info(
                    "Structured action item normalized title=%r assignee=%r due_date=%s "
                    "priority=%s status=%s confidence=%.2f source_type=%s",
                    redact(item.title), redact(item.assignee), item.due_date or "none",
                    item.priority or "none", item.status, item.confidence, item.source_type,
                )
                results.append(item)
    return _deduplicate(results)


def create_command(items):
    tasks = []
    for item in items:
        task = {"task_name": item.title, "status": "pending", "_source": {
            "type": item.source_type, "reference": item.source_reference,
            "confidence": item.confidence, "evidence": item.evidence,
        }}
        if item.assignee:
            task["assignee"] = item.assignee
        if item.due_date:
            task["due_date"] = item.due_date
        if item.priority:
            task["priority"] = item.priority
        tasks.append(task)
    return {"intent": "create", "tasks": tasks, "operations": [], "changes": []}


def workflow_command(item):
    """Translate one extracted action into the existing validated command shape."""
    source = {
        "type": item.source_type, "reference": item.source_reference,
        "confidence": item.confidence, "evidence": item.evidence,
    }
    if item.operation == "create":
        command = {"intent": "create", "task_name": item.title, "_source": source}
        if item.assignee:
            command["assignee"] = item.assignee
        if item.due_date:
            command["due_date"] = item.due_date
        if item.priority:
            command["priority"] = item.priority
        return command
    command = {
        "intent": item.operation, "task_name": item.title,
        "target_scope": "single", "_source": source,
    }
    if item.operation == "update":
        changes = []
        if item.assignee:
            changes.append({"field": "assignee", "value": item.assignee})
        if item.due_date:
            changes.append({"field": "due_date", "value": item.due_date})
        if item.priority:
            changes.append({"field": "priority", "value": item.priority})
        command["changes"] = changes
    return command


def serializable(items):
    return [asdict(item) for item in items]


def deserialize(values):
    return [ExtractedAction(**value) for value in values]
