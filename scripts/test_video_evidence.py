from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.scene_state import SceneState
from src.perception.types import SceneSnapshot, TrackEvent, TrackedObject
from src.perception.video.evidence import VideoEvidenceIndex
from src.perception.video.facts import VideoFacts
from src.perception.video.timeline import SemanticEvent, SemanticEventType, SemanticTimeline


def make_event(
    event_type: SemanticEventType,
    timestamp: float,
    frame_index: int | None,
    track_id: int,
    label: str,
    *,
    previous_position: tuple[float, float] | None = None,
    current_position: tuple[float, float] | None = None,
    confidence: float | None = 0.9,
) -> SemanticEvent:
    return SemanticEvent(
        event_type=event_type,
        timestamp=timestamp,
        frame_index=frame_index,
        track_id=track_id,
        label=label,
        previous_position=previous_position,
        current_position=current_position,
        confidence=confidence,
    )


def make_facts(events: list[SemanticEvent]) -> VideoFacts:
    timeline = SemanticTimeline(max(1, len(events)))
    grouped: dict[tuple[float, int | None], list[SemanticEvent]] = {}
    for event in events:
        grouped.setdefault((event.timestamp, event.frame_index), []).append(event)
    for (timestamp, frame_index), group in sorted(grouped.items(), key=lambda item: item[0][0]):
        track_events = []
        tracks = []
        for event in group:
            kind = {
                SemanticEventType.OBJECT_APPEARED: "appeared",
                SemanticEventType.OBJECT_REMAINED: "remained",
                SemanticEventType.OBJECT_MOVED: "moved",
                SemanticEventType.OBJECT_DISAPPEARED: "disappeared",
                SemanticEventType.OBJECT_REACQUIRED: "reacquired",
            }[event.event_type]
            track_events.append(
                TrackEvent(
                    kind=kind,
                    track_id=event.track_id,
                    label=event.label,
                    timestamp=event.timestamp,
                    previous_box=_box(event.previous_position),
                    current_box=_box(event.current_position),
                )
            )
            if event.current_position is not None and event.track_id is not None:
                x, y = event.current_position
                tracks.append(
                    TrackedObject(
                        track_id=event.track_id,
                        label=event.label,
                        confidence=event.confidence or 0.0,
                        bounding_box=(x, y, x, y),
                        normalized_horizontal=x,
                        normalized_vertical=y,
                        position_category="unknown",
                        vertical_position="unknown",
                        last_seen_timestamp=event.timestamp,
                        missed_frames=0,
                        state={
                            SemanticEventType.OBJECT_APPEARED: "visible",
                            SemanticEventType.OBJECT_REMAINED: "remained",
                            SemanticEventType.OBJECT_MOVED: "moved",
                            SemanticEventType.OBJECT_DISAPPEARED: "disappeared",
                            SemanticEventType.OBJECT_REACQUIRED: "reacquired",
                        }[event.event_type],
                    )
                )
        timeline.add_observation(
            SceneSnapshot(
                scene_state=SceneState(timestamp=timestamp),
                tracks=tuple(tracks),
                scene_changes=tuple(track_events),
                timestamp=timestamp,
                source_id="video:evidence-test",
            ),
            frame_index=frame_index,
        )
    return VideoFacts.from_timeline(timeline)


def _box(position: tuple[float, float] | None) -> tuple[float, float, float, float] | None:
    if position is None:
        return None
    x, y = position
    return (x, y, x, y)


def test_lifecycle_and_movement_evidence() -> None:
    appeared = make_event(
        SemanticEventType.OBJECT_APPEARED, 1.0, 10, 7, "person", current_position=(0.2, 0.5)
    )
    moved = make_event(
        SemanticEventType.OBJECT_MOVED,
        2.0,
        20,
        7,
        "person",
        previous_position=(0.2, 0.5),
        current_position=(0.7, 0.5),
    )
    disappeared = make_event(
        SemanticEventType.OBJECT_DISAPPEARED,
        3.0,
        30,
        7,
        "person",
        previous_position=(0.7, 0.5),
        confidence=None,
    )
    reacquired = make_event(
        SemanticEventType.OBJECT_REACQUIRED,
        4.0,
        40,
        7,
        "person",
        previous_position=(0.7, 0.5),
        current_position=(0.8, 0.5),
    )
    facts = make_facts([appeared, moved, disappeared, reacquired])
    index = VideoEvidenceIndex.from_facts(facts)
    appearance_evidence = index.by_event(appeared)[0]
    assert appearance_evidence.source_frame_index == 10
    assert appearance_evidence.previous_frame_index is None
    assert appearance_evidence.current_frame_index == 10
    assert appearance_evidence.current_position == (0.2, 0.5)
    assert appearance_evidence.confidence == 0.9

    movement_evidence = index.by_event(moved)[0]
    assert (movement_evidence.previous_frame_index, movement_evidence.previous_timestamp) == (10, 1.0)
    assert (movement_evidence.current_frame_index, movement_evidence.current_timestamp) == (20, 2.0)
    assert movement_evidence.previous_position == (0.2, 0.5)
    assert movement_evidence.current_position == (0.7, 0.5)

    disappearance_evidence = index.by_event(disappeared)[0]
    assert (disappearance_evidence.previous_frame_index, disappearance_evidence.previous_timestamp) == (20, 2.0)
    assert disappearance_evidence.current_frame_index == 30

    reacquisition_evidence = index.by_event(reacquired)[0]
    assert (reacquisition_evidence.previous_frame_index, reacquisition_evidence.previous_timestamp) == (30, 3.0)
    assert reacquisition_evidence.current_frame_index == 40
    print("APPEARANCE, MOVEMENT, DISAPPEARANCE, REACQUISITION EVIDENCE: PASS")


def test_queries_duplicate_labels_and_ordering() -> None:
    first = make_event(SemanticEventType.OBJECT_APPEARED, 1.0, 10, 1, "bottle")
    second = make_event(SemanticEventType.OBJECT_APPEARED, 1.0, 10, 2, "bottle")
    moved = make_event(SemanticEventType.OBJECT_MOVED, 2.0, 20, 1, "bottle")
    index = VideoEvidenceIndex.from_facts(make_facts([first, second, moved]))
    assert [item.track_id for item in index.by_track_id(1)] == [1, 1]
    assert [item.track_id for item in index.by_track_id(2)] == [2]
    assert [item.track_id for item in index.by_frame_index(10)] == [1, 2]
    assert [item.timestamp for item in index.events] == [1.0, 1.0, 2.0]
    print("DUPLICATE LABEL TRACK LOOKUPS AND CHRONOLOGICAL ORDER: PASS")


def test_optional_frame_indexes_and_no_fabricated_evidence() -> None:
    initial = make_event(SemanticEventType.OBJECT_APPEARED, 1.0, None, 3, "chair")
    moved_without_frame = make_event(
        SemanticEventType.OBJECT_MOVED,
        2.0,
        None,
        3,
        "chair",
        previous_position=(0.1, 0.2),
        current_position=(0.3, 0.4),
    )
    index = VideoEvidenceIndex.from_facts(make_facts([initial, moved_without_frame]))
    evidence = index.by_event(moved_without_frame)[0]
    assert evidence.source_frame_index is None
    assert evidence.current_frame_index is None
    assert evidence.previous_frame_index is None
    assert evidence.previous_timestamp == 1.0
    assert evidence.current_timestamp == 2.0
    assert evidence.previous_position == (0.1, 0.2)
    assert evidence.current_position == (0.3, 0.4)
    missing = make_event(SemanticEventType.OBJECT_APPEARED, 9.0, None, 99, "unknown")
    assert index.by_event(missing) == ()
    print("OPTIONAL FRAME INDEX AND NO FABRICATED EVIDENCE: PASS")


def test_bounded_history() -> None:
    events = [
        make_event(SemanticEventType.OBJECT_APPEARED, float(index), index, 5, "cup")
        for index in range(1, 5)
    ]
    index = VideoEvidenceIndex.from_facts(make_facts(events), history_size=2)
    assert len(index.events) == 2
    assert [item.timestamp for item in index.events] == [3.0, 4.0]
    assert index.by_frame_index(1) == ()
    print("BOUNDED EVIDENCE HISTORY: PASS")


def main() -> None:
    test_lifecycle_and_movement_evidence()
    test_queries_duplicate_labels_and_ordering()
    test_optional_frame_indexes_and_no_fabricated_evidence()
    test_bounded_history()
    print("VIDEO EVIDENCE INDEX: PASS")


if __name__ == "__main__":
    main()
