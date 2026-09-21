"""State fingerprints, duplicate detection and confirmation policy."""
import hashlib
import json
import re
import time
from difflib import SequenceMatcher

import slack_tools


CONFIRMATION_TTL_SECONDS = 600
DEFAULT_BULK_CONFIRMATION_THRESHOLD = 5


def _normalized_title(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def title_similarity(left, right):
    left_norm, right_norm = _normalized_title(left), _normalized_title(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    left_tokens, right_tokens = set(left_norm.split()), set(right_norm.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(jaccard, sequence)


def likely_duplicates(name, items, schema, assignee_ids=(), threshold=0.92):
    """Return only strong candidates; similar but distinct titles remain valid."""
    wanted_assignees = set(assignee_ids or ())
    matches = []
    for item in items:
        if slack_tools.extract_completed(item, schema):
            continue
        score = title_similarity(name, slack_tools.extract_item_name(item, schema))
        if score < threshold:
            continue
        existing_assignees = set(slack_tools.extract_assignee_ids(item, schema))
        # An explicitly different assignee weakens, but does not erase, an
        # exact-title duplicate. Near matches require compatible ownership.
        if score < 1.0 and wanted_assignees and existing_assignees != wanted_assignees:
            continue
        matches.append((score, item))
    return [item for _, item in sorted(matches, key=lambda pair: -pair[0])]


def item_fingerprint(item, schema):
    value = {
        "id": slack_tools.extract_item_id(item),
        "name": slack_tools.extract_item_name(item, schema),
        "assignees": slack_tools.extract_assignee_ids(item, schema),
        "due_date": slack_tools.extract_due_date(item, schema),
        "priority": slack_tools.extract_priority(item, schema),
        "completed": slack_tools.extract_completed(item, schema),
        # Preserve unknown dynamic cells so a proposal is invalidated when an
        # externally edited field changes, even if this application ignores it.
        "fields": item.get("fields") or item.get("cells") or [],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def snapshot_fingerprint(items, schema, item_ids=None):
    wanted = set(item_ids or ())
    selected = [item for item in items if not wanted or slack_tools.extract_item_id(item) in wanted]
    values = [(slack_tools.extract_item_id(item), item_fingerprint(item, schema)) for item in selected]
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def confirmation_required(intent, item_count, changes, threshold=DEFAULT_BULK_CONFIRMATION_THRESHOLD):
    if item_count < threshold:
        return False
    if intent == "delete":
        return True
    high_impact = {"assignee", "priority", "due_date", "completed", "status"}
    return intent in {"update", "complete", "reopen"} and any(
        change.get("field") in high_impact for change in changes)


def confirmation_is_fresh(confirmation, now=None):
    now = time.time() if now is None else now
    return bool(confirmation and confirmation.get("expires_at", 0) >= now)

