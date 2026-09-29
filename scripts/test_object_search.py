from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.scene_state import SceneObject, SceneState
from src.search.object_search import ObjectSearch, TargetLock, TargetState


def make_person(confidence: float, normalized_x: float, box: tuple[float, float, float, float]) -> SceneObject:
    position = "left" if normalized_x < 1 / 3 else "right" if normalized_x > 2 / 3 else "center"
    obj = SceneObject(
        label="person",
        confidence=confidence,
        center_x=(box[0] + box[2]) / 2,
        center_y=(box[1] + box[3]) / 2,
        normalized_horizontal=normalized_x,
        position_category=position,
    )
    obj.bounding_box = box
    return obj


def print_target(query: str, frame: str, target: TargetLock, lock_status: str) -> None:
    print(f"QUERY: {query} | FRAME: {frame}")
    print(f"TARGET LABEL: {target.label}")
    print(f"STATE: {target.state.value}")
    print(f"CONFIDENCE: {target.confidence}")
    print(f"POSITION: {target.position}")
    print(f"BOUNDING BOX: {target.bounding_box}")
    print(f"LOCK STATUS: {lock_status}")


def main() -> None:
    supported_queries = {
        "find person": "person",
        "find chair": "chair",
        "find bottle": "bottle",
        "find phone": "cell phone",
        "locate backpack": "backpack",
    }
    for query, expected_label in supported_queries.items():
        assert ObjectSearch.parse_query(query) == expected_label

    query = "find person"
    search = ObjectSearch(lost_after_frames=3)
    target_label = search.parse_query(query)
    assert target_label is not None

    frame_a = SceneState(
        objects=[
            make_person(0.91, 0.50, (230, 100, 410, 300)),
            make_person(0.76, 0.08, (0, 100, 100, 300)),
        ]
    )
    target = search.begin_search(target_label, frame_a)
    print_target(query, "A", target, "found; awaiting next-frame match")
    assert target.state is TargetState.FOUND
    assert target.confidence == 0.91

    frame_b = SceneState(objects=[make_person(0.89, 0.61, (300, 105, 480, 305))])
    target = search.update(frame_b)
    print_target(query, "B", target, "locked; matched shifted bounding box")
    assert target is not None and target.state is TargetState.LOCKED

    frame_c = SceneState(objects=[make_person(0.87, 0.72, (370, 110, 550, 310))])
    target = search.update(frame_c)
    print_target(query, "C", target, "locked; position updated")
    assert target is not None and target.state is TargetState.LOCKED
    assert target.position == "right"
    assert target.bounding_box == (370.0, 110.0, 550.0, 310.0)

    for miss_number in range(1, 4):
        target = search.update(SceneState())
        assert target is not None
        status = f"miss {miss_number}/3"
        if target.state is TargetState.LOST:
            status = "lost after 3 consecutive misses"
        print_target(query, f"D{miss_number}", target, status)
        expected_state = TargetState.LOST if miss_number == 3 else TargetState.LOCKED
        assert target.state is expected_state

    chair_query = "find chair"
    chair_label = ObjectSearch.parse_query(chair_query)
    assert chair_label is not None
    chair_search = ObjectSearch()
    chair_target = chair_search.begin_search(chair_label, SceneState())
    print_target(chair_query, "A", chair_target, "not found")
    assert chair_target.state is TargetState.SEARCHING
    assert chair_target.confidence is None

    print("OBJECT SEARCH LOGIC: PASS")


if __name__ == "__main__":
    main()
