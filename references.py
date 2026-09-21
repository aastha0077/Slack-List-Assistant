"""Pure reference grammar. Only complete noun phrases are references, never title substrings."""
import re
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Reference:
    kind: str
    positions: tuple = ()
    count: int = 0


class TargetType(str, Enum):
    SINGLE_ITEM = "single_item"
    MULTIPLE_ITEMS = "multiple_items"
    FILTERED_COLLECTION = "filtered_collection"
    ALL_APPLICABLE_ITEMS = "all_applicable_items"
    CONTEXTUAL_ITEMS = "contextual_items"


class TargetCardinality(str, Enum):
    SINGLE = "single"
    MULTIPLE = "multiple"
    COLLECTION = "collection"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ResolvedTargetSet:
    target_type: TargetType
    item_ids: tuple
    items: tuple


_ORDINALS = dict(zip(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split(),
    range(1, 21),
))
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_CARDINALS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_ORDINALS.update({"thirtieth": 30, "fortieth": 40, "fiftieth": 50,
                  "sixtieth": 60, "seventieth": 70, "eightieth": 80,
                  "ninetieth": 90, "hundredth": 100})


_COLLECTION_NOUN = r"(?:(?:action|todo|to-do)\s+)?(?:tasks?|items?|entries|work)"
_COLLECTION_QUANTIFIER = r"(?:all|every|each|entire|whole)"


def collection_scope(text):
    """Classify collection grammar independently from any operation wording."""
    value = str(text or "").strip().casefold().rstrip(".!?")
    explicit = re.search(
        rf"\b{_COLLECTION_QUANTIFIER}\b(?:\s+of)?(?:\s+(?:the|my|current|available|existing|pending|open|completed|overdue|due|high|low|priority)){{0,5}}\s+{_COLLECTION_NOUN}\b",
        value,
    )
    if explicit or re.search(r"\beverything\b", value):
        return "all_applicable"
    if re.search(r"\b(?:all|both)\b(?:\s+of)?\s+(?:these|those|them)\b", value):
        return "contextual"
    return None


def ordinal(text):
    text = text.strip().lower().replace("-", " ")
    if re.fullmatch(r"#?\d+(?:st|nd|rd|th)?", text):
        return int(re.sub(r"\D", "", text))
    if text in _ORDINALS:
        return _ORDINALS[text]
    words = text.split()
    if len(words) == 2 and words[0] in _TENS and words[1] in _ORDINALS:
        unit = _ORDINALS[words[1]]
        if unit < 10:
            return _TENS[words[0]] + unit
    return None


def parse_reference(text):
    text = str(text or "").strip().casefold().rstrip(".!?")
    # Quoted names are always names, including a task literally named "First".
    if any(c in text for c in '\"“”‘’'):
        return None
    text = re.sub(r"^(?:the\s+)", "", text)
    text = re.sub(r"\s+(?:on|from|in)\s+(?:that|this|the)\s+(?:displayed\s+)?list$", "", text)
    text = re.sub(r"\s+(?:from\s+)?above$", "", text)
    scope = collection_scope(text)
    if scope:
        return Reference("both" if scope == "contextual" and text.startswith("both") else "all")
    if re.fullmatch(r"(?:task|item|one)\s+(?:that\s+)?you\s+(?:just\s+)?(?:showed|created|updated|completed|mentioned)", text):
        return Reference("focus")
    if text in {"its", "their task"}:
        return Reference("focus")
    if text in {"those two", "these two", "the two"}:
        return Reference("both")
    if text in {"it", "that", "this", "that task", "this task", "that item", "this item",
                "that one", "this one", "the one above", "one above", "the one i mentioned",
                "task i mentioned", "the task i mentioned", "task we discussed",
                "the task we discussed", "the one we discussed"}:
        return Reference("focus")
    if text in {"them", "these", "those", "these tasks", "those tasks", "these items", "those items"}:
        return Reference("focus_set")
    if re.fullmatch(r"(?:all|both)(?:\s+(?:of\s+)?(?:them|these|those|the tasks|tasks|items|action items))?", text):
        return Reference("both" if text.startswith("both") else "all")
    if text == "everything":
        return Reference("all")
    if re.fullmatch(r"(?:previous|prior)(?:\s+(?:one|task|item|entry))?", text):
        return Reference("previous")
    if re.fullmatch(r"(?:next|following|another)(?:\s+(?:one|task|item|entry))?", text):
        return Reference("relative", (1,))
    if re.fullmatch(r"(?:(?:most\s+)?recent|latest|newest|last|final)(?:\s+(?:one|task|item|entry))?", text):
        return Reference("positions", (-1,))
    if re.fullmatch(r"(?:earliest|oldest)(?:\s+(?:one|task|item|entry))?", text):
        return Reference("positions", (1,))
    count = re.fullmatch(r"(?:first|last)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s+(?:tasks|items|ones))?", text)
    if count:
        amount = int(count[1]) if count[1].isdigit() else _CARDINALS[count[1]]
        return Reference("tail" if text.startswith("last") else "head", count=amount)
    text = re.sub(r"^(?:tasks?|items?|entr(?:y|ies)|numbers?|nos?\.)\s+", "", text)
    text = re.sub(r"\s+(?:ones?|tasks?|items?|entr(?:y|ies))$", "", text)
    text = re.sub(r"\bthe\s+", "", text)
    span = re.fullmatch(r"(#?\d+)(?:\s*(?:-|through|to)\s*)(#?\d+)", text)
    if span:
        start, end = ordinal(span[1]), ordinal(span[2])
        return Reference("positions", tuple(range(start, end + 1))) if end >= start else Reference("positions")
    parts = re.split(r"\s*(?:,\s*(?:and\s+)?|\band\b|&)\s*", text)
    positions = [-1 if part in {"last", "final"} else ordinal(part) for part in parts]
    if positions and all(p is not None for p in positions):
        return Reference("positions", tuple(dict.fromkeys(positions)))
    return None


def extract_contextual_reference(text):
    """Extract reference grammar embedded in a larger request without parsing intent."""
    value = str(text or "").casefold().replace("-", " ")
    # A quantity following first/last is one selection operator, even when a
    # member or status clause follows it ("first two of Sam's pending tasks").
    quantity = re.search(
        r"\b(first|last)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        value,
    )
    if quantity:
        amount = int(quantity.group(2)) if quantity.group(2).isdigit() else _CARDINALS[quantity.group(2)]
        return Reference("tail" if quantity.group(1) == "last" else "head", count=amount)
    # Keep alphanumeric title tokens intact: a title token ending in a digit is not
    # the ordinal 1. Standalone numeric ordinals remain separate tokens.
    tokens = re.findall(r"[a-z][a-z0-9]*|#?\d+(?:st|nd|rd|th)?", value)
    noun_positions = {i for i, token in enumerate(tokens)
                      if token in {"task", "tasks", "item", "items", "one", "ones", "entry", "entries"}}
    modifiers = {"and", "or", "the", "pending", "open", "completed", "overdue", "these", "those", "displayed"}
    def qualifies(index):
        for noun in sorted(position for position in noun_positions if position > index):
            between = tokens[index + 1:noun]
            if all(token in modifiers or ordinal(token) is not None or token in {"last", "final"}
                   for token in between):
                return True
            break
        return False
    positions = []
    for index, token in enumerate(tokens):
        position = ordinal(token)
        if position is not None and qualifies(index):
            positions.append(position)
        elif token in {"last", "final"} and qualifies(index):
            positions.append(-1)
    if positions:
        return Reference("positions", tuple(dict.fromkeys(positions)))
    for index, token in enumerate(tokens):
        if token in {"next", "following", "another"} and any(index < noun <= index + 2 for noun in noun_positions):
            return Reference("relative", (1,))
        if token in {"previous", "prior"} and any(index < noun <= index + 2 for noun in noun_positions):
            return Reference("previous")
    return None


def reference_from(parsed):
    """Accept the normalized grammar plus legacy parser fields at one boundary."""
    if parsed.get("literal_name"):
        return None
    value = parsed.get("reference")
    if isinstance(value, Reference):
        return value
    if isinstance(value, dict):
        return Reference(value["kind"], tuple(value.get("positions", ())), value.get("count", 0))
    if parsed.get("selection_numbers"):
        return Reference("positions", tuple(parsed["selection_numbers"]))
    if parsed.get("selection_count"):
        return Reference("head", count=int(parsed["selection_count"]))
    if parsed.get("selection_index") is not None:
        return Reference("positions", (int(parsed["selection_index"]),))
    selection = parsed.get("selection") or parsed.get("task_reference")
    if selection in {"__LAST__", "single"}:
        return Reference("focus")
    if selection:
        return parse_reference(selection)
    if parsed.get("task_name") == "__LAST__":
        return Reference("focus")
    return parse_reference(parsed.get("task_name"))


def requested_cardinality(parsed):
    """Return the user's target-count contract independently of candidate scope."""
    selection = parsed.get("target_selection") or {}
    if selection.get("mode") == "one":
        return TargetCardinality.SINGLE
    if selection.get("mode") == "many":
        return TargetCardinality.MULTIPLE
    if selection.get("mode") == "collection":
        return TargetCardinality.COLLECTION
    reference = reference_from(parsed)
    if reference:
        if reference.kind in {"focus", "previous", "relative"}:
            return TargetCardinality.SINGLE
        if reference.kind == "positions":
            return TargetCardinality.SINGLE if len(reference.positions) == 1 else TargetCardinality.MULTIPLE
        if reference.kind in {"head", "tail"} and reference.count == 1:
            return TargetCardinality.SINGLE
        if reference.kind in {"both", "head", "tail"}:
            return TargetCardinality.MULTIPLE
        if reference.kind in {"all", "focus_set"}:
            return TargetCardinality.COLLECTION
    scope = parsed.get("target_scope")
    if scope == "single" or parsed.get("limit") == 1:
        return TargetCardinality.SINGLE
    if scope == "multiple" or (isinstance(parsed.get("limit"), int) and parsed["limit"] > 1):
        return TargetCardinality.MULTIPLE
    if scope in {"filtered", "all_applicable", "contextual"}:
        return TargetCardinality.COLLECTION
    if parsed.get("task_name"):
        return TargetCardinality.SINGLE
    return TargetCardinality.AMBIGUOUS


def resolved_target_type(cardinality, source_scope=None):
    if cardinality == TargetCardinality.SINGLE:
        return TargetType.SINGLE_ITEM
    if cardinality == TargetCardinality.MULTIPLE:
        return TargetType.MULTIPLE_ITEMS
    if source_scope == "all_applicable":
        return TargetType.ALL_APPLICABLE_ITEMS
    if source_scope == "filtered":
        return TargetType.FILTERED_COLLECTION
    return TargetType.CONTEXTUAL_ITEMS


def enforce_cardinality(parsed, item_ids):
    """Fail closed when resolution expands beyond the user's requested shape."""
    cardinality = requested_cardinality(parsed)
    count = len(tuple(item_ids))
    if cardinality == TargetCardinality.SINGLE and count != 1:
        raise ValueError(f"I resolved {count} tasks for a singular request. Please clarify the exact task; no changes were made.")
    if cardinality == TargetCardinality.MULTIPLE and count < 1:
        raise ValueError("I couldn't resolve the requested tasks; no changes were made.")
    if cardinality == TargetCardinality.AMBIGUOUS and count != 1:
        raise ValueError("Please clarify whether you mean one task or a collection; no changes were made.")
    return cardinality


def select_ids(reference, displayed_ids, focus_ids=()):
    ids = list(displayed_ids)
    focus = list(focus_ids)
    if reference.kind == "focus":
        focus = list(focus_ids) or ids
        if len(focus) != 1:
            raise ValueError("Which task do you mean? Please use its displayed number or name.")
        return focus
    if reference.kind == "previous":
        if len(focus) == 1:
            return focus
        if ids:
            return [ids[-1]]
        raise ValueError("Which previous task do you mean?")
    if reference.kind == "both":
        if len(focus) == 2:
            return focus
        if len(ids) != 2:
            raise ValueError(f"I found {len(ids)} tasks. Which two tasks do you mean?")
        return ids
    if reference.kind == "focus_set":
        selected = focus or ids
        if not selected:
            raise ValueError("Which tasks do you mean? Please display or select them first.")
        return selected
    if reference.kind == "all":
        if not ids:
            raise ValueError("There are no tasks in that displayed list.")
        return ids
    if reference.kind in {"head", "tail"}:
        if not 1 <= reference.count <= len(ids):
            raise ValueError("That count is outside the displayed list.")
        return ids[:reference.count] if reference.kind == "head" else ids[-reference.count:]
    if reference.kind == "relative":
        if not ids or len(focus) != 1 or focus[0] not in ids:
            if reference.positions == (-1,) and ids and not focus:
                return [ids[-1]]
            raise ValueError("Which task should I use as the starting point for that reference?")
        position = ids.index(focus[0]) + reference.positions[0]
        if not 0 <= position < len(ids):
            raise ValueError("There is no task in that relative position in the displayed list.")
        return [ids[position]]
    if reference.kind != "positions" or not reference.positions:
        raise ValueError("Please specify a valid displayed position.")
    result = []
    for position in reference.positions:
        if not isinstance(position, int) or (position != -1 and not 1 <= position <= len(ids)) or not ids:
            raise ValueError("That position is outside the displayed list.")
        result.append(ids[-1] if position == -1 else ids[position - 1])
    return list(dict.fromkeys(result))
