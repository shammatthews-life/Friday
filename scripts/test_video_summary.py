from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.scene_state import SceneState
from src.perception.types import SceneSnapshot, TrackEvent, TrackedObject
from src.perception.video.evidence import VideoEvidenceIndex
from src.perception.video.facts import VideoFacts
from src.perception.video.summary import VideoEventSummarizer
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
    event_kind: str,
    track_id: int,
    label: str,
    x: float | None = None,
) -> None:
    state = {
        "appeared": "visible",
        "remained": "remained",
        "moved": "moved",
        "disappeared": "disappeared",
        "reacquired": "reacquired",
    }[event_kind]
    track = make_track(track_id, label, x if x is not None else 0.5, state)
    event = TrackEvent(
        kind=event_kind,
        track_id=track_id,
        label=label,
        timestamp=timestamp,
        previous_box=(x - 0.05, 0.4, x + 0.05, 0.6) if x is not None and event_kind != "appeared" else None,
        current_box=(x - 0.05, 0.4, x + 0.05, 0.6) if x is not None else None,
    )
    snapshot = SceneSnapshot(
        scene_state=SceneState(timestamp=timestamp),
        tracks=(track,),
        scene_changes=(event,),
        timestamp=timestamp,
        source_id="video:summary-test",
    )
    timeline.add_observation(snapshot, frame_index=frame_index)


def make_facts(timeline: SemanticTimeline) -> tuple[VideoFacts, VideoEvidenceIndex]:
    facts = VideoFacts.from_timeline(timeline)
    evidence = VideoEvidenceIndex.from_facts(facts)
    return facts, evidence


def test_presence_movement_and_disappearance_episode() -> None:
    timeline = SemanticTimeline()
    add(timeline, 1.0, 10, "appeared", 1, "person", 0.2)
    add(timeline, 2.0, 20, "remained", 1, "person", 0.2)
    add(timeline, 3.0, 30, "moved", 1, "person", 0.7)
    add(timeline, 4.0, 40, "disappeared", 1, "person", 0.7)
    facts, evidence = make_facts(timeline)
    episodes = VideoEventSummarizer().summarize(facts, evidence)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.episode_type == "activity"
    assert episode.start_timestamp == 1.0 and episode.end_timestamp == 4.0
    assert episode.event_types == (
        SemanticEventType.OBJECT_APPEARED,
        SemanticEventType.OBJECT_REMAINED,
        SemanticEventType.OBJECT_MOVED,
        SemanticEventType.OBJECT_DISAPPEARED,
    )
    assert len(episode.evidence) == 4
    assert episode.positions
    assert episode.confidences == (0.9, 0.9, 0.9, 0.9)
    print("APPEARANCE, CONTINUOUS PRESENCE, MOVEMENT, DISAPPEARANCE EPISODE: PASS")


def test_interruption_reacquisition_episode() -> None:
    timeline = SemanticTimeline()
    add(timeline, 1.0, 1, "appeared", 5, "cup", 0.3)
    add(timeline, 2.0, 2, "disappeared", 5, "cup", 0.3)
    add(timeline, 5.0, 5, "reacquired", 5, "cup", 0.6)
    add(timeline, 6.0, 6, "moved", 5, "cup", 0.8)
    facts, evidence = make_facts(timeline)
    episodes = VideoEventSummarizer().summarize(facts, evidence)
    assert len(episodes) == 1
    assert episodes[0].episode_type == "interrupted_reappearance"
    assert episodes[0].event_types == (
        SemanticEventType.OBJECT_APPEARED,
        SemanticEventType.OBJECT_DISAPPEARED,
        SemanticEventType.OBJECT_REACQUIRED,
        SemanticEventType.OBJECT_MOVED,
    )
    assert [item.timestamp for item in episodes[0].evidence] == [1.0, 2.0, 5.0, 6.0]
    print("DISAPPEARANCE AND REACQUISITION EPISODE WITH EVIDENCE: PASS")


def test_multiple_tracks_duplicate_labels_and_chronology() -> None:
    timeline = SemanticTimeline()
    first_track = make_track(1, "bottle", 0.2, "visible")
    second_track = make_track(2, "bottle", 0.8, "visible")
    timeline.add_observation(
        SceneSnapshot(
            scene_state=SceneState(timestamp=1.0),
            tracks=(first_track, second_track),
            scene_changes=(
                TrackEvent("appeared", 1, "bottle", 1.0, current_box=first_track.bounding_box),
                TrackEvent("appeared", 2, "bottle", 1.0, current_box=second_track.bounding_box),
            ),
            timestamp=1.0,
            source_id="video:summary-test",
        ),
        frame_index=1,
    )
    add(timeline, 3.0, 3, "disappeared", 1, "bottle", 0.2)
    add(timeline, 2.0, 2, "moved", 2, "bottle", 0.7)
    facts, evidence = make_facts(timeline)
    episodes = VideoEventSummarizer().summarize(facts, evidence)
    assert [episode.start_timestamp for episode in episodes] == [1.0, 1.0]
    assert [episode.track_id for episode in episodes] == [1, 2]
    assert episodes[0].label == episodes[1].label == "bottle"
    assert episodes[1].event_types == (
        SemanticEventType.OBJECT_APPEARED,
        SemanticEventType.OBJECT_MOVED,
    )
    assert episodes[1].incomplete
    print("MULTIPLE OBJECTS, DUPLICATE LABELS, CHRONOLOGICAL ORDER: PASS")


def test_incomplete_episodes_and_bounded_history() -> None:
    timeline = SemanticTimeline()
    add(timeline, 2.0, 2, "moved", 1, "chair", 0.4)
    add(timeline, 3.0, 3, "disappeared", 2, "table", 0.5)
    facts, evidence = make_facts(timeline)
    all_episodes = VideoEventSummarizer().summarize(facts, evidence)
    assert len(all_episodes) == 2
    activity = next(item for item in all_episodes if item.track_id == 1)
    assert activity.episode_type == "activity"
    assert activity.incomplete and activity.end_timestamp is None

    bounded = VideoEventSummarizer(history_size=1).summarize(facts, evidence)
    assert len(bounded) == 1 and bounded[0].track_id == 2
    table_episode = next(item for item in all_episodes if item.track_id == 2)
    assert table_episode.episode_type == "disappearance"
    assert table_episode.incomplete
    assert table_episode.end_timestamp == 3.0
    print("INCOMPLETE TIMELINE AND BOUNDED EPISODE HISTORY: PASS")


def test_appearance_only_and_empty_timeline() -> None:
    timeline = SemanticTimeline()
    add(timeline, 1.0, None, "appeared", 8, "person", 0.1)
    facts, evidence = make_facts(timeline)
    episode = VideoEventSummarizer().summarize(facts, evidence)[0]
    assert episode.episode_type == "presence"
    assert episode.incomplete
    assert episode.end_timestamp is None
    assert episode.events[0].frame_index is None
    assert episode.evidence[0].source_frame_index is None

    empty_facts, empty_evidence = make_facts(SemanticTimeline())
    assert VideoEventSummarizer().summarize(empty_facts, empty_evidence) == ()
    print("APPEARANCE-ONLY PARTIAL EPISODE AND EMPTY INPUT: PASS")


def main() -> None:
    test_presence_movement_and_disappearance_episode()
    test_interruption_reacquisition_episode()
    test_multiple_tracks_duplicate_labels_and_chronology()
    test_incomplete_episodes_and_bounded_history()
    test_appearance_only_and_empty_timeline()
    print("VIDEO EVENT SUMMARY: PASS")


if __name__ == "__main__":
    main()
