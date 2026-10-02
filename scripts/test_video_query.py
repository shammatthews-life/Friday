from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.scene_state import SceneState
from src.perception.types import SceneSnapshot, TrackEvent, TrackedObject
from src.perception.video.query import VideoQueryEngine
from src.perception.video.timeline import SemanticEventType, SemanticTimeline


def make_track(track_id: int, label: str, x: float, state: str) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        label=label,
        confidence=0.9,
        bounding_box=(x - 0.05, 0.4, x + 0.05, 0.6),
        normalized_horizontal=x,
        normalized_vertical=0.5,
        position_category="center",
        vertical_position="middle",
        last_seen_timestamp=0.0,
        missed_frames=0,
        state=state,
    )


def add(
    timeline: SemanticTimeline,
    timestamp: float,
    frame_index: int | None,
    kind: str,
    track_id: int,
    label: str,
    x: float | None = None,
    *,
    include_track: bool = True,
) -> None:
    state = {
        "appeared": "visible",
        "remained": "remained",
        "moved": "moved",
        "disappeared": "disappeared",
        "reacquired": "reacquired",
    }[kind]
    track = make_track(track_id, label, x if x is not None else 0.5, state) if include_track else None
    box = (x - 0.05, 0.4, x + 0.05, 0.6) if x is not None else None
    snapshot = SceneSnapshot(
        scene_state=SceneState(timestamp=timestamp),
        tracks=(track,) if track is not None else (),
        scene_changes=(
            TrackEvent(
                kind=kind,
                track_id=track_id,
                label=label,
                timestamp=timestamp,
                previous_box=box if kind in {"moved", "disappeared", "reacquired"} else None,
                current_box=box if kind in {"appeared", "moved", "reacquired"} else None,
            ),
        ),
        timestamp=timestamp,
        source_id="video:query-test",
    )
    timeline.add_observation(snapshot, frame_index=frame_index)


def make_engine() -> VideoQueryEngine:
    timeline = SemanticTimeline()
    add(timeline, 1.0, 10, "appeared", 1, "bottle", 0.2)
    add(timeline, 1.0, 11, "appeared", 2, "bottle", 0.8)
    add(timeline, 2.0, 20, "remained", 1, "bottle", 0.2)
    add(timeline, 3.0, 30, "moved", 1, "bottle", 0.6)
    add(timeline, 4.0, 40, "disappeared", 1, "bottle", 0.6, include_track=False)
    add(timeline, 5.0, 50, "reacquired", 1, "bottle", 0.7)
    add(timeline, 6.0, 60, "appeared", 3, "chair", None, include_track=False)
    return VideoQueryEngine(timeline)


def test_object_queries_and_identity() -> None:
    engine = make_engine()
    assert [item.track_id for item in engine.objects_present()] == [1, 2, 3]
    assert [item.track_id for item in engine.objects_appeared(label="BOTTLE")] == [1, 2]
    assert [item.track_id for item in engine.objects_disappeared()] == [1]
    assert [item.track_id for item in engine.objects_moved(label="bottle")] == [1]
    assert [item.track_id for item in engine.object_history(1)] == [1, 1, 1, 1, 1]
    assert engine.object_presence_duration(1).duration_seconds == 3.0
    assert engine.object_presence_duration(999) is None
    bottles = engine.objects_present(label="bottle")
    assert [item.track_id for item in bottles] == [1, 2]
    print("OBJECT SET QUERIES, DURATION, AND DUPLICATE LABEL IDENTITY: PASS")


def test_time_ranges_episodes_and_evidence() -> None:
    engine = make_engine()
    ranged = engine.events_in_time_range(2.0, 4.0, track_id=1)
    assert [event.timestamp for event in ranged] == [2.0, 3.0, 4.0]
    episodes = engine.episodes_in_time_range(2.5, 4.5, track_id=1)
    assert len(episodes) == 1
    assert episodes[0].track_id == 1
    movement = engine.objects_moved(track_id=1)[0]
    evidence = engine.evidence_for_event(movement)
    assert len(evidence) == 1
    assert evidence[0].current_frame_index == 30
    assert [event.timestamp for event in engine.timeline()] == [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    try:
        engine.events_in_time_range(4.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected reversed time range to fail")
    print("TIME RANGE, EPISODE, EVIDENCE, AND TIMELINE QUERIES: PASS")


def test_empty_results_and_optional_metadata() -> None:
    engine = make_engine()
    assert engine.objects_appeared(label="missing") == ()
    assert engine.events_in_time_range(100.0, 200.0) == ()
    assert engine.episodes_in_time_range(100.0, 200.0) == ()
    assert engine.objects_moved(track_id=999) == ()
    chair_event = next(event for event in engine.timeline() if event.label == "chair")
    assert chair_event.frame_index == 60
    assert chair_event.current_position is None
    assert chair_event.confidence is None
    assert engine.evidence_for_event(chair_event)[0].source_frame_index == 60

    empty = VideoQueryEngine(SemanticTimeline())
    assert empty.timeline() == ()
    assert empty.objects_present() == ()
    assert empty.objects_disappeared() == ()
    assert empty.object_presence_duration(99) is None
    print("EMPTY RESULTS AND MISSING OPTIONAL METADATA: PASS")


def main() -> None:
    test_object_queries_and_identity()
    test_time_ranges_episodes_and_evidence()
    test_empty_results_and_optional_metadata()
    print("VIDEO QUERY ENGINE: PASS")


if __name__ == "__main__":
    main()
