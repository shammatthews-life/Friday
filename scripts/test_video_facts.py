from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.scene_state import SceneState
from src.perception.types import SceneSnapshot, TrackEvent, TrackedObject
from src.perception.video.facts import VideoFacts
from src.perception.video.timeline import SemanticEvent, SemanticEventType, SemanticTimeline


def event(
    event_type: SemanticEventType,
    timestamp: float,
    track_id: int,
    label: str,
    *,
    frame_index: int | None = None,
    x: float | None = None,
    confidence: float | None = 0.9,
) -> SemanticEvent:
    position = (x, 0.5) if x is not None else None
    return SemanticEvent(
        event_type=event_type,
        timestamp=timestamp,
        frame_index=frame_index,
        track_id=track_id,
        label=label,
        current_position=position,
        confidence=confidence,
    )


def make_timeline(events: list[SemanticEvent], history_size: int = 256) -> SemanticTimeline:
    timeline = SemanticTimeline(history_size=history_size)
    grouped: dict[tuple[float, int | None], list[SemanticEvent]] = {}
    for item in events:
        grouped.setdefault((item.timestamp, item.frame_index), []).append(item)
    for (timestamp, frame_index), group in sorted(grouped.items(), key=lambda item: item[0][0]):
        track_states = {
            SemanticEventType.OBJECT_APPEARED: "visible",
            SemanticEventType.OBJECT_REMAINED: "remained",
            SemanticEventType.OBJECT_MOVED: "moved",
            SemanticEventType.OBJECT_DISAPPEARED: "disappeared",
            SemanticEventType.OBJECT_REACQUIRED: "reacquired",
        }
        tracks = tuple(
            TrackedObject(
                track_id=semantic.track_id,
                label=semantic.label,
                confidence=semantic.confidence,
                bounding_box=_position_box(semantic.current_position) or (0.0, 0.0, 0.0, 0.0),
                normalized_horizontal=(semantic.current_position or (0.0, 0.0))[0],
                normalized_vertical=(semantic.current_position or (0.0, 0.0))[1],
                position_category="unknown",
                vertical_position="unknown",
                last_seen_timestamp=timestamp,
                missed_frames=0,
                state=track_states[semantic.event_type],
            )
            for semantic in group
            if semantic.track_id is not None and semantic.confidence is not None
        )
        timeline.add_observation(
            SceneSnapshot(
                scene_state=SceneState(timestamp=timestamp),
                tracks=tracks,
                scene_changes=tuple(
                    TrackEvent(
                        kind={
                            SemanticEventType.OBJECT_APPEARED: "appeared",
                            SemanticEventType.OBJECT_MOVED: "moved",
                            SemanticEventType.OBJECT_DISAPPEARED: "disappeared",
                            SemanticEventType.OBJECT_REACQUIRED: "reacquired",
                            SemanticEventType.OBJECT_REMAINED: "remained",
                        }[semantic.event_type],
                        track_id=semantic.track_id,
                        label=semantic.label,
                        timestamp=semantic.timestamp,
                        previous_box=_position_box(semantic.previous_position),
                        current_box=_position_box(semantic.current_position),
                    )
                    for semantic in group
                    if semantic.track_id is not None
                ),
                timestamp=timestamp,
                source_id="video:facts-test",
            ),
            frame_index=frame_index,
        )
    return timeline


def _position_box(position: tuple[float, float] | None) -> tuple[float, float, float, float] | None:
    if position is None:
        return None
    x, y = position
    return (x, y, x, y)


def test_single_object_lifecycle_and_duration() -> None:
    timeline = make_timeline(
        [
            event(SemanticEventType.OBJECT_APPEARED, 1.0, 10, "person", frame_index=5, x=0.2),
            event(SemanticEventType.OBJECT_REMAINED, 2.0, 10, "person", frame_index=10, x=0.2),
            event(SemanticEventType.OBJECT_MOVED, 3.0, 10, "person", frame_index=15, x=0.6),
            event(SemanticEventType.OBJECT_DISAPPEARED, 5.0, 10, "person", frame_index=25),
            event(SemanticEventType.OBJECT_REACQUIRED, 8.0, 10, "person", frame_index=40, x=0.7),
            event(SemanticEventType.OBJECT_REMAINED, 9.0, 10, "person", frame_index=45, x=0.7),
        ]
    )
    facts = VideoFacts.from_timeline(timeline)
    person = facts.objects[0]
    assert person.track_id == 10
    assert person.first_seen_timestamp == 1.0
    assert person.last_seen_timestamp == 9.0
    assert person.presence_duration_seconds == 5.0
    assert person.appearance_count == 1
    assert person.reacquisition_count == 1
    assert person.is_present
    assert [item.event_type for item in facts.movement_events] == [SemanticEventType.OBJECT_MOVED]
    assert [item.track_id for item in facts.objects_present] == [10]
    assert [item.track_id for item in facts.objects_disappeared] == [10]
    history = facts.object_history(10)
    assert len(history) == 6
    assert history[2].frame_index == 15
    assert history[2].current_position == (0.6, 0.5)
    assert history[2].confidence == 0.9
    print("SINGLE OBJECT LIFECYCLE, MOVEMENT, DISAPPEARANCE, REACQUISITION, DURATION: PASS")


def test_multiple_objects_and_duplicate_labels_keep_track_identity() -> None:
    facts = VideoFacts.from_timeline(
        make_timeline(
            [
                event(SemanticEventType.OBJECT_APPEARED, 1.0, 1, "bottle", x=0.2),
                event(SemanticEventType.OBJECT_APPEARED, 1.0, 2, "bottle", x=0.8),
                event(SemanticEventType.OBJECT_DISAPPEARED, 2.0, 1, "bottle"),
                event(SemanticEventType.OBJECT_MOVED, 3.0, 2, "bottle", x=0.7),
                event(SemanticEventType.OBJECT_APPEARED, 4.0, 3, "chair", x=0.5),
            ]
        )
    )
    bottles = [item for item in facts.objects if item.label == "bottle"]
    assert [item.track_id for item in bottles] == [1, 2]
    assert [item.track_id for item in facts.objects_present] == [2, 3]
    assert [item.track_id for item in facts.objects_appeared] == [1, 2, 3]
    assert [item.timestamp for item in facts.event_timeline] == [1.0, 1.0, 2.0, 3.0, 4.0]
    assert [item.track_id for item in facts.movement_events] == [2]
    print("MULTIPLE OBJECTS, DUPLICATE LABEL TRACK IDENTITY, CHRONOLOGY: PASS")


def test_empty_timeline_and_bounded_history() -> None:
    empty = VideoFacts.from_timeline(SemanticTimeline())
    assert empty.events == ()
    assert empty.objects == ()
    assert empty.objects_present == ()
    assert empty.to_dict() == {"events": [], "objects": []}

    timeline = make_timeline(
        [
            event(SemanticEventType.OBJECT_APPEARED, 1.0, 7, "cup"),
            event(SemanticEventType.OBJECT_REMAINED, 2.0, 7, "cup"),
            event(SemanticEventType.OBJECT_MOVED, 3.0, 7, "cup"),
            event(SemanticEventType.OBJECT_DISAPPEARED, 4.0, 7, "cup"),
        ]
    )
    bounded = VideoFacts.from_timeline(timeline, history_size=2)
    assert [item.event_type for item in bounded.event_timeline] == [
        SemanticEventType.OBJECT_MOVED,
        SemanticEventType.OBJECT_DISAPPEARED,
    ]
    assert bounded.objects[0].first_seen_timestamp == 3.0
    assert bounded.objects[0].last_seen_timestamp == 3.0
    assert bounded.objects[0].is_present is False
    print("EMPTY TIMELINE AND EXPLICITLY BOUNDED FACT HISTORY: PASS")


def test_to_dict_preserves_structured_metadata() -> None:
    facts = VideoFacts.from_timeline(
        make_timeline(
            [event(SemanticEventType.OBJECT_APPEARED, 2.5, 4, "chair", frame_index=75, x=0.4)]
        )
    )
    serialized = facts.to_dict()
    serialized_event = serialized["events"][0]
    assert serialized_event == {
        "event_type": "OBJECT_APPEARED",
        "timestamp": 2.5,
        "frame_index": 75,
        "track_id": 4,
        "label": "chair",
        "previous_position": None,
        "current_position": [0.4, 0.5],
        "confidence": 0.9,
    }
    print("MACHINE-READABLE FACT SERIALIZATION: PASS")


def main() -> None:
    test_single_object_lifecycle_and_duration()
    test_multiple_objects_and_duplicate_labels_keep_track_identity()
    test_empty_timeline_and_bounded_history()
    test_to_dict_preserves_structured_metadata()
    print("VIDEO FACT EXTRACTION: PASS")


if __name__ == "__main__":
    main()
