from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.scene_memory import SceneMemory
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


def labels(objects) -> list[str]:
    return [f"{obj.label} #{obj.track_id}" for obj in objects]


def report(memory: SceneMemory, update_number: int) -> None:
    visible = memory.get_visible_objects()
    newly_noticed = memory.get_newly_noticed_objects()
    disappeared = memory.get_recently_disappeared_objects()
    visible_counts = {label: memory.count(label) for label in ("person", "clock", "bottle")}
    print(f"UPDATE: {update_number}")
    print(f"VISIBLE OBJECTS: {labels(visible)}")
    print(f"COUNTS: {visible_counts}")
    print(f"NEWLY NOTICED: {labels(newly_noticed)}")
    print(f"RECENTLY DISAPPEARED: {[f'{obj.label} #{obj.track_id}' for obj in disappeared]}")


def main() -> None:
    memory = SceneMemory(retention_seconds=4.0, miss_limit=2, position_tolerance=0.25)

    memory.update(
        make_scene(
            1.0,
            make_object("Person", 0.91, "center", 0.50, 1.0),
            make_object("clock", 0.88, "left", 0.15, 1.0),
        )
    )
    report(memory, 1)
    assert memory.count("PERSON") == 1
    assert {obj.label for obj in memory.get_newly_noticed_objects()} == {"person", "clock"}

    memory.update(
        make_scene(
            2.0,
            make_object("person", 0.90, "center", 0.52, 2.0),
            make_object("clock", 0.87, "left", 0.16, 2.0),
        )
    )
    report(memory, 2)
    assert memory.get_newly_noticed_objects() == []

    memory.update(
        make_scene(
            3.0,
            make_object("person", 0.89, "center", 0.53, 3.0),
            make_object("clock", 0.86, "left", 0.17, 3.0),
            make_object("bottle", 0.84, "right", 0.82, 3.0),
        )
    )
    report(memory, 3)
    assert [obj.label for obj in memory.get_newly_noticed_objects()] == ["bottle"]

    memory.update(make_scene(4.0, make_object("person", 0.88, "center", 0.54, 4.0)))
    report(memory, 4)
    assert memory.find_objects("clock")
    assert memory.get_recently_disappeared_objects() == []

    memory.update(make_scene(5.0, make_object("person", 0.87, "center", 0.55, 5.0)))
    report(memory, 5)
    disappeared_labels = [obj.label for obj in memory.get_recently_disappeared_objects()]
    assert "clock" in disappeared_labels
    assert memory.was_recently_seen("clock")

    memory.update(
        make_scene(
            6.0,
            make_object("person", 0.91, "left", 0.30, 6.0),
            make_object("person", 0.83, "right", 0.76, 6.0),
        )
    )
    report(memory, 6)
    assert memory.count("person") == 2

    memory.update(make_scene(11.0))
    report(memory, 7)
    assert memory.get_visible_objects() == []
    assert memory.find_objects("person") == []
    assert not memory.was_recently_seen("person")

    print("SCENE MEMORY LOGIC: PASS")


if __name__ == "__main__":
    main()
