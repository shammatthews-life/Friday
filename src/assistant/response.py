from __future__ import annotations

from collections import Counter

from src.assistant.intent import IntentType, ParsedIntent
from src.core.scene_memory import Disappearance, RememberedObject, SceneMemory
from src.core.scene_state import SceneState
from src.search.object_search import TargetLock, TargetState


def _matches(scene: SceneState, label: str | None):
    if not label:
        return scene.objects
    return [obj for obj in scene.objects if obj.label.lower() == label.lower()]


def respond_to_intent(
    parsed: ParsedIntent, scene: SceneState, scene_memory: SceneMemory | None = None
) -> str:
    matches = _matches(scene, parsed.object_label)
    if parsed.intent is IntentType.WHAT_AROUND:
        if not scene.objects:
            return "I can see nothing clearly right now."
        counts = Counter(obj.label for obj in scene.objects)
        parts = [f"{count} {label}" if count > 1 else f"a {label}" for label, count in counts.items()]
        return "I can see " + ", ".join(parts) + "."
    if parsed.intent is IntentType.IS_OBJECT_PRESENT:
        label = parsed.object_label or "that object"
        return f"Yes, there is a {label}." if matches else f"No, I do not see a {label}."
    if parsed.intent is IntentType.WHERE_OBJECT:
        label = parsed.object_label or "that object"
        return f"The {label} is on your {matches[0].position_category}." if matches else f"I do not see a {label}."
    if parsed.intent is IntentType.COUNT_OBJECT:
        label = parsed.object_label
        count = scene_memory.count(label) if scene_memory is not None and label else len(matches)
        noun = label or "object"
        if noun == "person" and count != 1:
            noun = "people"
        suffix = "" if count == 1 or noun.endswith("s") or noun == "people" else "s"
        return f"I can see {count} {noun}{suffix}."
    if parsed.intent is IntentType.SCENE_CHANGES:
        return "I did not notice any new or missing objects."
    if parsed.intent is IntentType.RECENTLY_SEEN:
        return "I have not seen any objects recently."
    return "I am not sure what you are asking yet."


def respond_to_scene_changes(
    newly_noticed: list[RememberedObject], disappeared: list[Disappearance]
) -> str:
    changes: list[str] = []
    changes.extend(f"I noticed a {obj.label}." for obj in newly_noticed)
    changes.extend(f"The {obj.label} disappeared from view." for obj in disappeared)
    return " ".join(changes) if changes else "I did not notice any new or missing objects."


def respond_to_recently_seen(
    scene_memory: SceneMemory, label: str | None = None
) -> str:
    if label:
        if scene_memory.was_recently_seen(label):
            return f"I recently saw a {label}."
        return f"I have not seen a {label} recently."

    recent: dict[int, str] = {
        obj.track_id: obj.label for obj in scene_memory.get_visible_objects()
    }
    recent.update(
        {obj.track_id: obj.label for obj in scene_memory.get_recently_disappeared_objects()}
    )
    counts: dict[str, int] = {}
    for object_label in recent.values():
        counts[object_label] = counts.get(object_label, 0) + 1
    if not counts:
        return "I have not seen any objects recently."

    parts = []
    for object_label, count in counts.items():
        if object_label == "person" and count > 1:
            parts.append(f"{count} people")
        elif count > 1:
            parts.append(f"{count} {object_label}s")
        else:
            parts.append(f"a {object_label}")
    if len(parts) == 1:
        summary = parts[0]
    elif len(parts) == 2:
        summary = " and ".join(parts)
    else:
        summary = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return f"I recently saw {summary}."


def respond_to_search(target: TargetLock) -> str:
    label = target.label
    if target.state is TargetState.SEARCHING:
        return f"I do not see a {label}."
    if target.state is TargetState.LOST:
        return f"I lost track of the {label}."

    confidence = f" (confidence {target.confidence:.2f})" if target.confidence is not None else ""
    if target.position == "center":
        location = "in the center"
    elif target.position in {"left", "right"}:
        location = f"on your {target.position}"
    else:
        location = "in view"
    if target.state is TargetState.LOCKED:
        return f"The {label} is still {location}{confidence}."
    return f"I found the {label} {location}{confidence}."
