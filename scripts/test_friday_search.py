from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.friday import FridayAssistant
from src.assistant.intent import IntentType, parse_intent
from src.core.scene_state import SceneObject, SceneState
from src.search.object_search import TargetState


def make_person(confidence: float, normalized_x: float, box: tuple[float, float, float, float]) -> SceneObject:
    position = "left" if normalized_x < 1 / 3 else "right" if normalized_x > 2 / 3 else "center"
    person = SceneObject(
        label="person",
        confidence=confidence,
        center_x=(box[0] + box[2]) / 2,
        center_y=(box[1] + box[3]) / 2,
        normalized_horizontal=normalized_x,
        position_category=position,
    )
    person.bounding_box = box
    return person


def main() -> None:
    assert "ultralytics" not in sys.modules
    assert "cv2" not in sys.modules

    supported_queries = {
        "find person": "person",
        "find chair": "chair",
        "find bottle": "bottle",
        "find phone": "phone",
        "locate backpack": "backpack",
    }
    for query, label in supported_queries.items():
        parsed = parse_intent(query)
        assert parsed.intent is IntentType.SEARCH_OBJECT
        assert parsed.object_label == label

    friday = FridayAssistant()
    assert friday.model is None
    found_response = friday.respond(
        "find person",
        SceneState(objects=[make_person(0.91, 0.50, (230, 100, 410, 300))]),
    )
    print(f"FOUND RESPONSE: {found_response}")
    assert found_response == "I found the person in the center (confidence 0.91)."
    assert friday.object_search.target is not None
    assert friday.object_search.target.state is TargetState.FOUND

    locked_response = friday.respond(
        "find person",
        SceneState(objects=[make_person(0.89, 0.72, (300, 105, 480, 305))]),
    )
    print(f"LOCKED RESPONSE: {locked_response}")
    assert locked_response == "The person is still on your right (confidence 0.89)."
    assert friday.object_search.target is not None
    assert friday.object_search.target.state is TargetState.LOCKED
    assert friday.object_search.target.position == "right"

    for _ in range(3):
        lost_response = friday.respond("find person", SceneState())
    print(f"LOST RESPONSE: {lost_response}")
    assert lost_response == "I lost track of the person."
    assert friday.object_search.target is not None
    assert friday.object_search.target.state is TargetState.LOST

    absent_response = friday.respond("find chair", SceneState())
    print(f"ABSENT-OBJECT RESPONSE: {absent_response}")
    assert absent_response == "I do not see a chair."
    print("FRIDAY SEARCH INTEGRATION: PASS")


if __name__ == "__main__":
    main()