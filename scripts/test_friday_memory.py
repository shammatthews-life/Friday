from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.friday import FridayAssistant
from src.assistant.intent import IntentType, parse_intent
from src.core.scene_state import SceneObject, SceneState


def make_object(
    label: str,
    confidence: float,
    position: str,
    normalized_horizontal: float,
    timestamp: float,
) -> SceneObject:
    return SceneObject(
        label=label,
        confidence=confidence,
        position_category=position,
        normalized_horizontal=normalized_horizontal,
        timestamp=timestamp,
    )


def make_scene(timestamp: float, *objects: SceneObject) -> SceneState:
    return SceneState(objects=list(objects), timestamp=timestamp)


def main() -> None:
    assert "ultralytics" not in sys.modules
    assert "cv2" not in sys.modules

    assert parse_intent("What changed?").intent is IntentType.SCENE_CHANGES
    assert parse_intent("Did anything appear?").intent is IntentType.SCENE_CHANGES
    assert parse_intent("Did anything disappear?").intent is IntentType.SCENE_CHANGES
    assert parse_intent("What did you see recently?").intent is IntentType.RECENTLY_SEEN
    assert parse_intent("Was there a bottle?").object_label == "bottle"

    friday = FridayAssistant()
    scene_one = make_scene(
        1.0,
        make_object("person", 0.91, "center", 0.50, 1.0),
        make_object("clock", 0.88, "left", 0.15, 1.0),
    )
    recently_seen = friday.respond("What did you see recently?", scene_one)
    print(f"RECENTLY SEEN: {recently_seen}")
    assert recently_seen == "I recently saw a person and a clock."

    scene_two = make_scene(
        2.0,
        make_object("person", 0.90, "center", 0.51, 2.0),
        make_object("clock", 0.87, "left", 0.16, 2.0),
    )
    unchanged = friday.respond("What changed?", scene_two)
    print(f"UNCHANGED SCENE: {unchanged}")
    assert unchanged == "I did not notice any new or missing objects."

    scene_three = make_scene(
        3.0,
        make_object("person", 0.89, "center", 0.52, 3.0),
        make_object("clock", 0.86, "left", 0.17, 3.0),
        make_object("bottle", 0.84, "right", 0.82, 3.0),
    )
    new_object = friday.respond("What changed?", scene_three)
    print(f"NEW OBJECT: {new_object}")
    assert new_object == "I noticed a bottle."
    bottle_history = friday.respond("Was there a bottle?", scene_three)
    print(f"RECENT BOTTLE QUERY: {bottle_history}")
    assert bottle_history == "I recently saw a bottle."

    scene_four = make_scene(
        4.0,
        make_object("person", 0.88, "center", 0.53, 4.0),
        make_object("bottle", 0.83, "right", 0.81, 4.0),
    )
    disappeared = friday.respond("What changed?", scene_four)
    print(f"DISAPPEARED OBJECT: {disappeared}")
    assert disappeared == "The clock disappeared from view."

    scene_five = make_scene(
        5.0,
        make_object("person", 0.92, "left", 0.20, 5.0),
        make_object("person", 0.82, "right", 0.80, 5.0),
        make_object("bottle", 0.81, "right", 0.80, 5.0),
    )
    count_response = friday.respond("How many people are there?", scene_five)
    print(f"DUPLICATE-PERSON COUNT: {count_response}")
    assert count_response == "I can see 2 people."

    search_response = friday.respond("find person", scene_five)
    print(f"EXISTING SEARCH: {search_response}")
    assert search_response == "I found the person on your left (confidence 0.92)."
    print("FRIDAY SCENE MEMORY INTEGRATION: PASS")


if __name__ == "__main__":
    main()
