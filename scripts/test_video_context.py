from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from test_video_session import MockVideoSource, make_frame, make_pipeline
from src.perception.types import PerceptionFrame
from src.perception.video.context import VideoContextBuilder, VideoContextLimits
from src.perception.video.knowledge import VideoKnowledge
from src.perception.video.ocr import OCRDetection, VideoTextExtractor, VideoTextObservation
from src.perception.video.session import VideoAnalysisSession
from src.perception.video.text_history import VideoTextHistory


class SequenceOCRBackend:
    def __init__(self, texts: tuple[str | None, ...]) -> None:
        self.texts = texts
        self.index = 0

    def extract(self, image: np.ndarray) -> list[OCRDetection]:
        text = self.texts[self.index]
        self.index += 1
        if text is None:
            return []
        if text == "UNLOCATED":
            return [OCRDetection(text, None, None)]
        return [OCRDetection(text, 0.9, (1.0, 2.0, 20.0, 12.0))]


class MinimalSource:
    def __init__(self, frames: list[PerceptionFrame]) -> None:
        self.frames = frames

    def __iter__(self):
        return iter(self.frames)

    def close(self) -> None:
        pass


def test_session_exposes_knowledge_and_empty_context() -> None:
    empty = VideoAnalysisSession(MockVideoSource([]), make_pipeline()).run()
    assert empty.knowledge.facts is empty.facts
    assert empty.knowledge.episodes is empty.episodes
    assert empty.knowledge.evidence_index is empty.evidence
    assert empty.knowledge.text_history is empty.text_history
    empty_context = VideoContextBuilder(empty.knowledge).build()
    assert empty_context["visual_objects"] == []
    assert empty_context["visual_events"] == []
    assert empty_context["text_observations"] == []
    assert empty_context["evidence_references"] == []
    assert empty.to_dict()["knowledge"] == empty.knowledge.to_dict()
    print("SESSION KNOWLEDGE INTEGRATION AND EMPTY VIDEO: PASS")


def make_complete_result() -> object:
    frames = [make_frame(index) for index in range(7)]
    backend = SequenceOCRBackend(
        ("EXIT", "ROOM 204", "EXIT", "UNLOCATED", None, None, None)
    )
    return VideoAnalysisSession(
        MockVideoSource(frames),
        make_pipeline(),
        text_extractor=VideoTextExtractor(backend),
    ).run()


def test_partial_session_remains_grounded() -> None:
    frames = [
        make_frame(0, np.zeros((48, 100, 3), dtype=np.uint8)),
        make_frame(1),
    ]
    result = VideoAnalysisSession(MockVideoSource(frames), make_pipeline()).run()
    assert result.status == "partial"
    assert result.knowledge.facts is result.facts
    context = VideoContextBuilder(result.knowledge).build()
    assert context["visual_events"]
    assert all(item["frame_index"] == 1 for item in context["visual_events"])
    print("PARTIAL VIDEO CONTEXT: PASS")


def test_fusion_queries_and_chronological_grounding() -> None:
    result = make_complete_result()
    knowledge = result.knowledge
    assert knowledge.facts is result.facts
    assert knowledge.episodes is result.episodes
    assert knowledge.evidence_index is result.evidence
    assert knowledge.text_history is result.text_history

    context = VideoContextBuilder(knowledge).build()
    assert len(context["visual_objects"]) == len(result.facts.objects)
    assert len(context["text_observations"]) == len(result.text_observations)
    assert [item["timestamp"] for item in context["visual_events"]] == sorted(
        item["timestamp"] for item in context["visual_events"]
    )
    assert [item["timestamp"] for item in context["text_observations"]] == sorted(
        item["timestamp"] for item in context["text_observations"]
    )
    bottle_objects = [
        item for item in context["visual_objects"] if item["label"] == "bottle"
    ]
    assert len(bottle_objects) == 2
    assert len({item["track_id"] for item in bottle_objects}) == 2
    assert context["evidence_references"] == result.evidence.to_dict()["events"]
    assert all(
        reference["source_frame_index"] == reference["current_frame_index"]
        for reference in context["evidence_references"]
    )
    print("VISUAL/TEXT FUSION, DUPLICATE LABELS, AND EVIDENCE: PASS")


def test_time_object_and_text_contexts() -> None:
    result = make_complete_result()
    builder = VideoContextBuilder(result.knowledge)
    time_context = builder.for_time_range(0.1, 0.2)
    assert all(0.1 <= item["timestamp"] <= 0.2 for item in time_context["visual_events"])
    assert all(
        0.1 <= item["timestamp"] <= 0.2 for item in time_context["text_observations"]
    )
    assert all(item["end_timestamp"] is None for item in time_context["episodes"])
    assert time_context["time_ranges"]["requested"] == {
        "start_timestamp": 0.1,
        "end_timestamp": 0.2,
    }

    bottle_id = next(
        item.track_id for item in result.facts.objects if item.label == "bottle"
    )
    object_context = builder.for_object(bottle_id)
    assert object_context["visual_objects"]
    assert all(item["track_id"] == bottle_id for item in object_context["visual_objects"])
    assert all(item["track_id"] == bottle_id for item in object_context["visual_events"])
    assert object_context["text_observations"] == []

    text_context = builder.for_text(" exit ")
    assert text_context["visual_events"] == []
    assert text_context["visual_objects"] == []
    assert text_context["text_observations"]
    assert all(item["text"].casefold() == "exit" for item in text_context["text_observations"])
    assert all(
        item["normalized_text"] == "exit"
        for item in text_context["text_history"]["events"]
    )
    print("TIME-RANGE, OBJECT, AND TEXT CONTEXT FILTERS: PASS")


def test_time_range_separates_text_reappearance_episodes() -> None:
    result = make_complete_result()
    history = VideoTextHistory()
    first = VideoTextObservation("EXIT", 0.9, 0.0, 0, (1.0, 2.0, 20.0, 12.0), "video:a")
    second = VideoTextObservation("EXIT", 0.8, 3.1, 31, (1.0, 2.0, 20.0, 12.0), "video:a")
    history.add_frame(
        timestamp=0.0,
        frame_index=0,
        source_id="video:a",
        observations=(first,),
    )
    history.add_frame(
        timestamp=3.0,
        frame_index=30,
        source_id="video:a",
        observations=(),
    )
    history.add_frame(
        timestamp=3.1,
        frame_index=31,
        source_id="video:a",
        observations=(second,),
    )
    knowledge = VideoKnowledge.from_layers(
        facts=result.facts,
        episodes=result.episodes,
        evidence_index=result.evidence,
        text_history=history,
        text_observations=(first, second),
    )
    selected = VideoContextBuilder(knowledge).for_time_range(3.05, 3.2)
    episodes = selected["text_history"]["episodes"]
    assert len(episodes) == 1
    assert episodes[0]["first_timestamp"] == 3.1
    assert episodes[0]["frame_indexes"] == [31]
    print("TEXT REAPPEARANCE EPISODES STAY SEPARATE IN TIME CONTEXT: PASS")


def test_context_limits_and_missing_metadata_are_explicit() -> None:
    result = make_complete_result()
    limits = VideoContextLimits(
        max_objects=1,
        max_events=1,
        max_episodes=1,
        max_text_observations=1,
        max_text_history_episodes=1,
        max_text_history_events=1,
        max_evidence_references=1,
        max_nested_events=1,
        max_nested_evidence=1,
        max_nested_text_observations=1,
    )
    bounded = VideoContextBuilder(result.knowledge, limits=limits).build()
    assert len(bounded["visual_objects"]) <= 1
    assert len(bounded["visual_events"]) <= 1
    assert len(bounded["episodes"]) <= 1
    assert len(bounded["text_observations"]) <= 1
    assert len(bounded["evidence_references"]) <= 1
    assert bounded["truncated"]["visual_objects"] > 0
    assert bounded["truncated"]["visual_events"] > 0
    assert all(len(item["events"]) <= 1 for item in bounded["visual_objects"])
    assert all(len(item["events"]) <= 1 for item in bounded["episodes"])
    assert all(len(item["evidence"]) <= 1 for item in bounded["episodes"])
    assert all(
        len(item["observations"]) <= 1
        for item in bounded["text_history"]["episodes"]
    )

    frame = make_frame(0)
    frame = PerceptionFrame(
        image=frame.image,
        timestamp=frame.timestamp,
        source_id="",
        frame_index=None,
    )
    missing = VideoAnalysisSession(
        MinimalSource([frame]),
        make_pipeline(),
        text_extractor=VideoTextExtractor(SequenceOCRBackend(("UNLOCATED",))),
    ).run()
    missing_context = VideoContextBuilder(missing.knowledge).build()
    observation = missing_context["text_observations"][0]
    assert missing_context["video_metadata"]["input"]["source_id"] is None
    assert observation["frame_index"] is None
    assert observation["source_id"] is None
    assert observation["confidence"] is None
    assert observation["bbox"] is None
    visual_event = missing_context["visual_events"][0]
    assert "bbox" not in visual_event
    assert "source_id" not in visual_event
    assert visual_event["current_position"] is None or len(
        visual_event["current_position"]
    ) == 2
    print("BOUNDED CONTEXT AND ABSENT FIELDS ARE NOT FABRICATED: PASS")


def main() -> None:
    test_session_exposes_knowledge_and_empty_context()
    test_partial_session_remains_grounded()
    test_fusion_queries_and_chronological_grounding()
    test_time_object_and_text_contexts()
    test_time_range_separates_text_reappearance_episodes()
    test_context_limits_and_missing_metadata_are_explicit()
    print("VIDEO GROUNDED CONTEXT: PASS")


if __name__ == "__main__":
    main()
