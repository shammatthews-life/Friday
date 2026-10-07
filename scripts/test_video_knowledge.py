from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.video.evidence import VideoEvidenceIndex
from src.perception.video.facts import ObjectVideoFacts, VideoFacts
from src.perception.video.knowledge import VideoKnowledge
from src.perception.video.ocr import VideoTextObservation
from src.perception.video.summary import VideoEpisode
from src.perception.video.text_history import VideoTextHistory
from src.perception.video.timeline import SemanticEvent, SemanticEventType


def make_event(
    timestamp: float,
    frame_index: int | None,
    track_id: int | None,
    label: str,
    *,
    confidence: float | None = 0.9,
) -> SemanticEvent:
    return SemanticEvent(
        event_type=SemanticEventType.OBJECT_APPEARED,
        timestamp=timestamp,
        frame_index=frame_index,
        track_id=track_id,
        label=label,
        previous_position=None,
        current_position=(0.25, 0.75),
        confidence=confidence,
    )


def make_knowledge() -> VideoKnowledge:
    events = (
        make_event(2.0, 20, 8, "bottle"),
        make_event(1.0, 10, 3, "bottle"),
    )
    facts = VideoFacts(
        events=events,
        objects=(
            ObjectVideoFacts(8, "bottle", 2.0, 2.0, 0.0, 1, 0, True, (events[0],)),
            ObjectVideoFacts(3, "bottle", 1.0, 1.0, 0.0, 1, 0, True, (events[1],)),
        ),
    )
    evidence = VideoEvidenceIndex.from_facts(facts)
    episodes = tuple(
        VideoEpisode(
            episode_type="presence",
            track_id=event.track_id,
            label=event.label,
            start_timestamp=event.timestamp,
            end_timestamp=None,
            events=(event,),
            positions=((0.25, 0.75),),
            confidences=(0.9,),
            evidence=evidence.by_event(event),
            incomplete=True,
        )
        for event in events
    )
    observations = (
        VideoTextObservation("EXIT", 0.95, 1.5, 15, (1.0, 2.0, 30.0, 14.0), "video:a"),
        VideoTextObservation("ROOM 204", None, 2.5, None, None, "video:a"),
    )
    history = VideoTextHistory()
    history.add_frame(
        timestamp=1.5,
        frame_index=15,
        source_id="video:a",
        observations=(observations[0],),
    )
    history.add_frame(
        timestamp=2.5,
        frame_index=None,
        source_id="video:a",
        observations=(observations[1],),
    )
    return VideoKnowledge(
        facts=facts,
        episodes=episodes,
        evidence_index=evidence,
        text_history=history,
        text_observations=observations,
        video_metadata={"input": {"source_id": "video:a", "fps": None}},
    )


def test_visual_and_text_fusion_and_grounding() -> None:
    knowledge = make_knowledge()
    result = knowledge.to_dict()
    assert len(result["visual"]["objects"]) == 2
    assert len(result["visual"]["episodes"]) == 2
    assert len(result["visual"]["events"]) == 2
    assert len(result["text"]["observations"]) == 2
    assert result["video_metadata"]["input"]["fps"] is None
    first_visual = result["visual"]["events"][0]
    assert first_visual["timestamp"] == 1.0
    assert first_visual["frame_index"] == 10
    assert first_visual["track_id"] == 3
    assert first_visual["label"] == "bottle"
    assert first_visual["confidence"] == 0.9
    assert first_visual["current_position"] == [0.25, 0.75]
    assert len(result["visual"]["evidence_references"]) == 2
    assert result["visual"]["evidence_references"][0]["source_frame_index"] == 10
    print("VISUAL AND TEXT FUSION PRESERVES GROUNDING AND EVIDENCE: PASS")


def test_duplicate_labels_and_queries() -> None:
    knowledge = make_knowledge()
    assert knowledge.object_information(3).track_id == 3
    assert knowledge.object_information(8).track_id == 8
    assert knowledge.object_information(99) is None
    ordered = knowledge.events_in_time_range(1.0, 2.0)
    assert [(event.timestamp, event.frame_index) for event in ordered] == [
        (1.0, 10),
        (2.0, 20),
    ]
    assert len(knowledge.evidence_references(track_id=8)) == 1
    text_range = knowledge.text_information_in_time_range(1.0, 2.0)
    assert [item["text"] for item in text_range["observations"]] == ["EXIT"]
    assert [item["frame_index"] for item in text_range["events"]] == [15]
    print("DUPLICATE LABELS, TEMPORAL ACCESS, AND EVIDENCE QUERIES: PASS")


def test_empty_sections_and_missing_metadata() -> None:
    empty = VideoKnowledge(
        facts=VideoFacts(events=(), objects=()),
        episodes=(),
        evidence_index=VideoEvidenceIndex(),
        text_history=VideoTextHistory(),
    )
    serialized = empty.to_dict()
    assert serialized["visual"] == {
        "objects": [],
        "events": [],
        "episodes": [],
        "evidence_references": [],
    }
    assert serialized["text"] == {"observations": [], "history": {"episodes": [], "events": []}}
    assert serialized["video_metadata"] == {}

    knowledge = make_knowledge()
    item = knowledge.to_dict()["text"]["observations"][1]
    assert item["confidence"] is None
    assert item["frame_index"] is None
    assert item["bbox"] is None
    print("EMPTY SECTIONS AND MISSING OPTIONAL METADATA: PASS")


def test_from_analysis_preserves_session_metadata() -> None:
    knowledge = make_knowledge()
    result = SimpleNamespace(
        status="completed",
        input_metadata=SimpleNamespace(
            to_dict=lambda: {"source_id": "video:a", "fps": 30.0}
        ),
        sampled_frame_count=4,
        processed_frame_count=4,
        invalid_perception_frame_count=0,
        facts=knowledge.facts,
        episodes=knowledge.episodes,
        evidence=knowledge.evidence_index,
        text_history=knowledge.text_history,
        text_observations=knowledge.text_observations,
    )
    serialized = VideoKnowledge.from_analysis(result).to_dict()
    assert serialized["video_metadata"] == {
        "status": "completed",
        "input": {"source_id": "video:a", "fps": 30.0},
        "sampled_frame_count": 4,
        "processed_frame_count": 4,
        "invalid_perception_frame_count": 0,
    }
    print("SESSION RESULT ADAPTER PRESERVES VIDEO METADATA: PASS")


def main() -> None:
    test_visual_and_text_fusion_and_grounding()
    test_duplicate_labels_and_queries()
    test_empty_sections_and_missing_metadata()
    test_from_analysis_preserves_session_metadata()
    print("VIDEO KNOWLEDGE: PASS")


if __name__ == "__main__":
    main()
