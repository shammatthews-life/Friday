from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class IntentType(str, Enum):
    SEARCH_OBJECT = "SEARCH_OBJECT"
    SCENE_CHANGES = "SCENE_CHANGES"
    RECENTLY_SEEN = "RECENTLY_SEEN"
    WHAT_AROUND = "WHAT_AROUND"
    IS_OBJECT_PRESENT = "IS_OBJECT_PRESENT"
    WHERE_OBJECT = "WHERE_OBJECT"
    COUNT_OBJECT = "COUNT_OBJECT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ParsedIntent:
    intent: IntentType
    object_label: str | None = None


def _capture(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    value = re.sub(r"\s+", " ", match.group(1).strip().lower())
    value = re.sub(r"^(?:a|an|the)\s+", "", value)
    if value == "people":
        value = "person"
    return value or None


def parse_intent(text: str) -> ParsedIntent:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    what_about = _capture(normalized, r"what about (.+?)[?.!]*$")
    if what_about:
        return ParsedIntent(IntentType.WHERE_OBJECT, what_about)
    can_find_again = _capture(normalized, r"can you find (.+?) again[?.!]*$")
    if can_find_again:
        return ParsedIntent(IntentType.SEARCH_OBJECT, can_find_again)
    if re.fullmatch(r"is (?:it|that|that object|the same one) still there\??", normalized):
        reference = re.search(r"is (it|that|that object|the same one) still there", normalized)
        return ParsedIntent(IntentType.IS_OBJECT_PRESENT, reference.group(1) if reference else None)
    if re.fullmatch(r"what changed\??|did anything (?:appear|disappear)\??", normalized):
        return ParsedIntent(IntentType.SCENE_CHANGES)
    if re.fullmatch(r"what did you see recently\??", normalized):
        return ParsedIntent(IntentType.RECENTLY_SEEN)
    recently_seen = _capture(normalized, r"was there (.+?)[?.!]*$")
    if recently_seen:
        return ParsedIntent(IntentType.RECENTLY_SEEN, recently_seen)
    if re.match(r"^(?:find|locate)\s+", normalized):
        label = _capture(normalized, r"(?:find|locate)\s+(.+?)[?.!]*$")
        if label:
            return ParsedIntent(IntentType.SEARCH_OBJECT, label)
    if re.search(r"\b(what is around me|what do you see|what's around me)\b", normalized):
        return ParsedIntent(IntentType.WHAT_AROUND)
    if re.search(r"\bhow many\b", normalized):
        label = _capture(normalized, r"how many (.+?)(?: are there| do you see|\?|$)")
        return ParsedIntent(IntentType.COUNT_OBJECT, None if label in {"object", "objects", "thing", "things"} else label)
    if re.search(r"\b(where is|where are|where can i find)\b", normalized):
        return ParsedIntent(
            IntentType.WHERE_OBJECT,
            _capture(normalized, r"(?:where is|where are|where can i find) (.+?)(?:\s+now)?[?.!]*$"),
        )
    if re.search(r"\b(is there|do you see|are there)\b", normalized):
        return ParsedIntent(IntentType.IS_OBJECT_PRESENT, _capture(normalized, r"(?:is there|do you see|are there) (.+?)(?:\?|$)"))
    return ParsedIntent(IntentType.UNKNOWN)
