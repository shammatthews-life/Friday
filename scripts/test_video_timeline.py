from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.scene_state import SceneState
from src.perception.types import SceneSnapshot, TrackEvent, TrackedObject
from src.perception.video.timeline import SemanticEventType, SemanticTimeline


def make_track(
    track_id: int,
    label: str,
    x: float,
    *,
    state: str = "visible",
    confidence: float = 0.9,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        label=label,
        confidence=confidence,
        bounding_box=(x - 0.05, 0.4, x + 0.05, 0.6),
        normalized_horizontal=x,
        normalized_vertical=0.5,
        position_category="center",
        vertical_position="middle",
        last_seen_timestamp=0.0,
        missed_frames=0,
        state=state,
    )


def make_snapshot(
    timestamp: float,
    tracks: tuple[TrackedObject, ...],
    *events: TrackEvent,
) -> SceneSnapshot:
    return SceneSnapshot(
        scene_state=SceneState(timestamp=timestamp),
        tracks=tracks,
        scene_changes=events,
        timestamp=timestamp,
        source_id="video:timeline-test",
    )


def test_object_lifecycle_and_duplicate_remained_suppression() -> None:
    timeline = SemanticTimeline()
    appeared = make_track(7, "person", 0.2)
    appeared_event = TrackEvent("appeared", 7, "person", 1.0, current_box=appeared.bounding_box)
    emitted = timeline.add_observation(make_snapshot(1.0, (appeared,), appeared_event), frame_index=10)
    assert [event.event_type for event in emitted] == [SemanticEventType.OBJECT_APPEARED]
    assert emitted[0].track_id == 7
    assert emitted[0].current_position == (0.2, 0.5)
    assert emitted[0].confidence == 0.9

    remained = make_track(7, "person", 0.2, state="remained")
    emitted = timeline.add_observation(make_snapshot(2.0, (remained,)), frame_index=20)
    assert [event.event_type for event in emitted] == [SemanticEventType.OBJECT_REMAINED]
    assert emitted[0].previous_position == emitted[0].current_position == (0.2, 0.5)
    assert timeline.add_observation(make_snapshot(3.0, (remained,)), frame_index=30) == ()
    duplicate_frame = timeline.add_observation(make_snapshot(3.0, (remained,)), frame_index=30)
    assert duplicate_frame == ()

    moved = make_track(7, "person", 0.6, state="moved")
    moved_event = TrackEvent(
        "moved",
        7,
        "person",
        4.0,
        previous_box=remained.bounding_box,
        current_box=moved.bounding_box,
    )
    emitted = timeline.add_observation(make_snapshot(4.0, (moved,), moved_event), frame_index=40)
    assert [event.event_type for event in emitted] == [SemanticEventType.OBJECT_MOVED]
    assert emitted[0].previous_position == (0.2, 0.5)
    assert emitted[0].current_position == (0.6, 0.5)

    disappeared_event = TrackEvent(
        "disappeared",
        7,
        "person",
        5.0,
        previous_box=moved.bounding_box,
    )
    emitted = timeline.add_observation(
        make_snapshot(5.0, (make_track(7, "person", 0.6, state="disappeared"),), disappeared_event),
        frame_index=50,
    )
    assert [event.event_type for event in emitted] == [SemanticEventType.OBJECT_DISAPPEARED]

    reacquired = make_track(7, "person", 0.7, state="reacquired")
    reacquired_event = TrackEvent(
        "reacquired",
        7,
        "person",
        6.0,
        previous_box=moved.bounding_box,
        current_box=reacquired.bounding_box,
    )
    emitted = timeline.add_observation(
        make_snapshot(6.0, (reacquired,), reacquired_event),
        frame_index=60,
    )
    assert [event.event_type for event in emitted] == [SemanticEventType.OBJECT_REACQUIRED]
    assert emitted[0].track_id == 7
    assert emitted[0].previous_position == (0.6, 0.5)
    assert emitted[0].current_position == (0.7, 0.5)
    print("OBJECT APPEARED/REMAINED/MOVED/DISAPPEARED/REACQUIRED: PASS")


def test_multiple_tracks_duplicate_labels_and_ordering() -> None:
    timeline = SemanticTimeline()
    first = make_track(1, "bottle", 0.2)
    second = make_track(2, "bottle", 0.8)
    events = (
        TrackEvent("appeared", 1, "bottle", 2.0, current_box=first.bounding_box),
        TrackEvent("appeared", 2, "bottle", 2.0, current_box=second.bounding_box),
    )
    timeline.add_observation(make_snapshot(2.0, (first, second), *events), frame_index=20)
    timeline.add_observation(
        make_snapshot(
            1.0,
            (make_track(3, "chair", 0.5),),
            TrackEvent("appeared", 3, "chair", 1.0),
        ),
        frame_index=10,
    )
    assert [event.timestamp for event in timeline.events] == [1.0, 2.0, 2.0]
    duplicate_label_ids = [
        event.track_id for event in timeline.events if event.label == "bottle"
    ]
    assert duplicate_label_ids == [1, 2]
    print("MULTIPLE TRACKS, DUPLICATE LABEL IDENTITY, CHRONOLOGICAL ORDER: PASS")


def test_history_is_bounded() -> None:
    timeline = SemanticTimeline(history_size=2)
    for index in range(3):
        track = make_track(index + 1, "object", 0.5)
        event = TrackEvent("appeared", track.track_id, track.label, float(index))
        timeline.add_observation(
            make_snapshot(float(index), (track,), event),
            frame_index=index,
        )
    assert len(timeline.events) == 2
    assert [event.track_id for event in timeline.events] == [2, 3]
    print("BOUNDED TIMELINE HISTORY: PASS")


def main() -> None:
    test_object_lifecycle_and_duplicate_remained_suppression()
    test_multiple_tracks_duplicate_labels_and_ordering()
    test_history_is_bounded()
    print("SEMANTIC VIDEO TIMELINE: PASS")


if __name__ == "__main__":
    main()
