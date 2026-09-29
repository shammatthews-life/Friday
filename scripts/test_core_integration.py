from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.friday import FridayAssistant
from src.core.scene_state import SceneObject, SceneState
from src.navigation.guidance import GuidanceEngine
from src.safety.safety_engine import AlertPriority, SafetyEngine, SafetyInput, SafetyState
from src.search.object_search import ObjectSearch, TargetState


def make_object(label: str, confidence: float, position: str, x: float, timestamp: float) -> SceneObject:
    return SceneObject(
        label=label,
        confidence=confidence,
        normalized_horizontal=x,
        position_category=position,
        timestamp=timestamp,
    )


def make_scene(timestamp: float, *objects: SceneObject) -> SceneState:
    return SceneState(objects=list(objects), timestamp=timestamp)


def check_contains(response: str, *phrases: str) -> None:
    normalized = response.lower()
    assert all(phrase.lower() in normalized for phrase in phrases), response


def main() -> None:
    friday = FridayAssistant()
    print("MODEL/HARDWARE: not invoked; all scene data is deterministic mock input")

    scene_one = make_scene(
        1.0,
        make_object("person", 0.91, "center", 0.50, 1.0),
        make_object("chair", 0.88, "right", 0.82, 1.0),
        make_object("bottle", 0.84, "left", 0.18, 1.0),
    )
    print(f"1. SCENESTATE: {[(obj.label, obj.confidence, obj.position_category) for obj in scene_one.objects]}")

    around_response = friday.respond("What is around me?", scene_one)
    check_contains(around_response, "person", "chair", "bottle")
    recent_response = friday.respond("What did you see recently?", scene_one)
    check_contains(recent_response, "person", "chair", "bottle")
    print(f"2. SCENEMEMORY: visible={[(obj.label, obj.position) for obj in friday.scene_memory.get_visible_objects()]}")
    print(f"3. FRIDAY SCENE 1: around={around_response!r}; recent={recent_response!r}")

    scene_two = make_scene(
        2.0,
        make_object("person", 0.91, "right", 0.65, 2.0),
        make_object("chair", 0.88, "right", 0.82, 2.0),
        make_object("bottle", 0.84, "left", 0.18, 2.0),
    )
    where_person = friday.respond("Where is the person?", scene_two)
    check_contains(where_person, "person", "right")
    initial_search_response = friday.respond("Find the person.", scene_two)
    target = friday.object_search.target
    assert target is not None and target.state in {TargetState.FOUND, TargetState.LOCKED}
    print(f"4. OBJECTSEARCH: after initial search={target.state.value}; response={initial_search_response!r}")

    bottle_context_response = friday.respond("What about the bottle?", scene_two)
    check_contains(bottle_context_response, "bottle", "left")
    bottle_followup = friday.respond("Where is it?", scene_two)
    check_contains(bottle_followup, "bottle", "left")
    print(
        f"3. FRIDAY SCENE 2: where_person={where_person!r}; "
        f"bottle_context={bottle_context_response!r}; followup={bottle_followup!r}"
    )

    scene_three = make_scene(
        3.0,
        make_object("person", 0.90, "right", 0.66, 3.0),
        make_object("bottle", 0.83, "left", 0.19, 3.0),
    )
    changed_response = friday.respond("What changed?", scene_three)
    check_contains(changed_response, "chair", "disappeared")
    target = friday.object_search.target
    assert target is not None
    if target.state is TargetState.FOUND:
        target = friday.object_search.update(scene_three)
    assert target is not None and target.state is TargetState.LOCKED
    print(f"2. SCENEMEMORY: disappeared={[(obj.label, obj.track_id) for obj in friday.scene_memory.get_recently_disappeared_objects()]}")
    print(f"3. FRIDAY SCENE 3: {changed_response!r}")
    print(f"4. OBJECTSEARCH: scene 3 state={target.state.value}")

    for frame_index in range(1, friday.object_search.lost_after_frames + 1):
        target = friday.object_search.update(make_scene(3.0 + frame_index / 10))
        assert target is not None
        print(f"4. OBJECTSEARCH: target-loss frame {frame_index}={target.state.value}")
    assert target is not None and target.state is TargetState.LOST

    guidance = GuidanceEngine(initial_target_state=TargetState.LOCKED)
    guidance_result = guidance.update(target_state=target.state)
    assert guidance_result.target_state is TargetState.LOST
    assert guidance_result.guidance_state.value == "TARGET_LOST"
    print(f"5. GUIDANCE: {guidance_result}")

    safety = SafetyEngine(initial_state=SafetyState.CAUTION)
    near_input = SafetyInput(
        obstacle_present=True,
        relative_depth="relatively near",
        confidence=0.90,
        target_is_tracked=True,
        timestamp=10.0,
    )
    stop_result = safety.evaluate(near_input)
    assert (stop_result.state, stop_result.priority) == (SafetyState.STOP, AlertPriority.HIGH)
    print(f"6. SAFETY NEAR: {stop_result}")

    clear_input = SafetyInput(
        obstacle_present=False,
        relative_depth="unknown",
        confidence=None,
        target_is_tracked=True,
        timestamp=10.1,
    )
    intermediate_result = safety.evaluate(clear_input)
    assert intermediate_result.state is SafetyState.CAUTION
    print(f"6. SAFETY CLEAR STEP 1: {intermediate_result}")
    final_clear_input = SafetyInput(
        obstacle_present=False,
        relative_depth="unknown",
        confidence=None,
        target_is_tracked=True,
        timestamp=10.2,
    )
    clear_result = safety.evaluate(final_clear_input)
    assert (clear_result.state, clear_result.priority) == (SafetyState.CLEAR, AlertPriority.NORMAL)
    print(f"6. SAFETY CLEAR STEP 2: {clear_result}")

    assert "ultralytics" not in sys.modules
    assert "cv2" not in sys.modules
    print("INTEGRATION = PASS")


if __name__ == "__main__":
    main()
