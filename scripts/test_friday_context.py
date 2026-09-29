from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.friday import FridayAssistant
from src.assistant.intent import IntentType, parse_intent
from src.core.scene_state import SceneObject, SceneState


def make_object(label: str, confidence: float, position: str, x: float) -> SceneObject:
    return SceneObject(
        label=label,
        confidence=confidence,
        position_category=position,
        normalized_horizontal=x,
        timestamp=1.0,
    )


def main() -> None:
    assert "ultralytics" not in sys.modules
    assert "cv2" not in sys.modules

    scene = SceneState(
        objects=[
            make_object("person", 0.91, "right", 0.82),
            make_object("bottle", 0.87, "left", 0.18),
        ],
        timestamp=1.0,
    )
    friday = FridayAssistant()

    person_response = friday.respond("Where is the person?", scene)
    print(f"EXPLICIT OBJECT: {person_response}")
    assert person_response == "The person is on your right."
    assert friday.conversation_context.last_relevant_object_label == "person"
    assert friday.conversation_context.last_relevant_position == "right"

    moved_scene = SceneState(
        objects=[
            make_object("person", 0.89, "left", 0.18),
            make_object("bottle", 0.87, "right", 0.82),
        ],
        timestamp=2.0,
    )
    it_response = friday.respond("Where is it?", moved_scene)
    print(f"IT FOLLOW-UP: {it_response}")
    assert it_response == "The person is on your left."
    assert friday.conversation_context.last_relevant_position == "left"

    depth_response = friday.respond("How far is it?", moved_scene)
    print(f"DEPTH FOLLOW-UP: {depth_response}")
    assert depth_response == "I can see the person, but I cannot estimate its relative depth."

    bottle_response = friday.respond("What about the bottle?", moved_scene)
    print(f"CONTEXT SWITCH: {bottle_response}")
    assert bottle_response == "The bottle is on your right."
    assert friday.conversation_context.last_relevant_object_label == "bottle"

    bottle_it_response = friday.respond("Where is it now?", moved_scene)
    print(f"BOTTLE FOLLOW-UP: {bottle_it_response}")
    assert bottle_it_response == "The bottle is on your right."

    assert parse_intent("Is it still there?").intent is IntentType.IS_OBJECT_PRESENT
    assert parse_intent("Can you find it again?").intent is IntentType.SEARCH_OBJECT
    found_response = friday.respond("Can you find it again?", moved_scene)
    print(f"CONTEXTUAL SEARCH: {found_response}")
    assert found_response == "I found the bottle on your right (confidence 0.87)."

    ambiguous_friday = FridayAssistant()
    scene_summary = ambiguous_friday.respond("What do you see?", scene)
    assert scene_summary == "I can see a person, a bottle."
    ambiguous_response = ambiguous_friday.respond("Where is it?", scene)
    print(f"AMBIGUOUS CASE: {ambiguous_response}")
    assert ambiguous_response == "I am not sure which object you mean."

    no_context_friday = FridayAssistant()
    no_context_response = no_context_friday.respond("Where is it?", scene)
    print(f"NO-CONTEXT CASE: {no_context_response}")
    assert no_context_response == "I don't know which object you mean."

    search_friday = FridayAssistant()
    search_response = search_friday.respond("find person", scene)
    print(f"EXISTING SEARCH: {search_response}")
    assert search_response == "I found the person on your right (confidence 0.91)."

    memory_friday = FridayAssistant()
    initial_memory_response = memory_friday.respond("What did you see recently?", scene)
    assert initial_memory_response == "I recently saw a person and a bottle."
    repeated_scene = SceneState(
        objects=[
            make_object("person", 0.91, "right", 0.82),
            make_object("bottle", 0.87, "left", 0.18),
        ],
        timestamp=2.0,
    )
    no_changes_response = memory_friday.respond("What changed?", repeated_scene)
    assert no_changes_response == "I did not notice any new or missing objects."
    print("EXISTING SEARCH AND MEMORY: PASS")
    print("FRIDAY CONTEXT TEST: PASS")


if __name__ == "__main__":
    main()
