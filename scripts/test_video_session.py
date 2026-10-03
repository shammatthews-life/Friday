from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.pipeline import PerceptionPipeline
from src.perception.types import Detection, PerceptionFrame
from src.perception.video.session import VideoAnalysisSession
from src.perception.video.video_source import VideoInputError
from src.perception.spatial.relations import horizontal_region, vertical_region


class MockVideoSource:
    def __init__(self, frames: list[PerceptionFrame]) -> None:
        self.frames = frames
        self.video_path = Path("mock_video.mp4")
        self.fps = 10.0
        self.decoded_frames = len(frames)
        self.yielded_frames = 0
        self.skipped_invalid_frames = 0
        self.closed = False

    def __iter__(self):
        for frame in self.frames:
            self.yielded_frames += 1
            yield frame

    def close(self) -> None:
        self.closed = True


class UnreadableVideoSource:
    video_path = Path("missing.mp4")
    fps = None
    decoded_frames = 0
    yielded_frames = 0
    skipped_invalid_frames = 0
    last_error = None

    def __iter__(self):
        raise VideoInputError("mock source is unreadable")

    def close(self) -> None:
        pass


class MockDetector:
    def __init__(self, schedule: dict[int, list[tuple[str, tuple[float, float, float, float]]]]) -> None:
        self.schedule = schedule

    def detect(self, frame: PerceptionFrame) -> list[Detection]:
        index = frame.frame_index or 0
        detections = []
        for label, box in self.schedule.get(index, []):
            left, top, right, bottom = box
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            normalized_x = center_x / frame.width
            normalized_y = center_y / frame.height
            detections.append(
                Detection(
                    label=label,
                    confidence=0.9,
                    bounding_box=box,
                    center_x=center_x,
                    center_y=center_y,
                    normalized_horizontal=normalized_x,
                    normalized_vertical=normalized_y,
                    position_category=horizontal_region(normalized_x),
                    vertical_position=vertical_region(normalized_y),
                )
            )
        return detections


def make_frame(index: int, image: np.ndarray | None = None) -> PerceptionFrame:
    if image is None:
        base = np.indices((48, 100)).sum(axis=0).astype(np.uint8)
        image = (30 + base % 180)[:, :, None].repeat(3, axis=2)
    return PerceptionFrame(
        image=image,
        timestamp=index / 10.0,
        source_id="video:mock_video.mp4",
        frame_index=index,
    )


def make_pipeline() -> PerceptionPipeline:
    def box(center_x: int) -> tuple[float, float, float, float]:
        return (center_x - 6, 10, center_x + 6, 34)

    schedule = {
        0: [("person", box(20)), ("bottle", box(65)), ("bottle", box(85))],
        1: [("person", box(35)), ("bottle", box(65)), ("bottle", box(85))],
        2: [("person", box(35)), ("bottle", box(65)), ("bottle", box(85))],
        3: [("bottle", box(65)), ("bottle", box(85))],
        4: [("bottle", box(65)), ("bottle", box(85))],
        5: [("bottle", box(65)), ("bottle", box(85))],
        6: [("person", box(40)), ("bottle", box(65)), ("bottle", box(85))],
    }
    return PerceptionPipeline(MockDetector(schedule), depth_provider=None)


def test_empty_and_unreadable_sources() -> None:
    empty_source = MockVideoSource([])
    empty = VideoAnalysisSession(empty_source, make_pipeline()).run()
    assert empty.status == "empty"
    assert empty.sampled_frame_count == empty.processed_frame_count == 0
    assert empty.timeline.events == empty.facts.events == empty.evidence.events == ()
    assert empty.episodes == ()
    assert empty.queries.timeline() == ()
    assert empty_source.closed

    unreadable = VideoAnalysisSession(UnreadableVideoSource(), make_pipeline()).run()
    assert unreadable.status == "unreadable"
    assert unreadable.sampled_frame_count == unreadable.processed_frame_count == 0
    assert unreadable.errors and "unreadable" in unreadable.errors[0]
    print("EMPTY AND UNREADABLE VIDEO SOURCES: PASS")


def test_end_to_end_tracking_facts_evidence_and_queries() -> None:
    source = MockVideoSource([make_frame(index) for index in range(7)])
    result = VideoAnalysisSession(source, make_pipeline()).run()
    assert result.status == "completed"
    assert result.sampled_frame_count == result.processed_frame_count == 7
    assert source.closed

    person_events = [event for event in result.timeline.events if event.label == "person"]
    event_types = [event.event_type.value for event in person_events]
    assert "OBJECT_APPEARED" in event_types
    assert "OBJECT_MOVED" in event_types
    assert "OBJECT_DISAPPEARED" in event_types
    assert "OBJECT_REACQUIRED" in event_types
    assert [event.frame_index for event in person_events] == [0, 1, 2, 5, 6]
    assert [event.timestamp for event in person_events] == [0.0, 0.1, 0.2, 0.5, 0.6]

    bottle_facts = [item for item in result.facts.objects if item.label == "bottle"]
    assert len(bottle_facts) == 2
    assert len({item.track_id for item in bottle_facts}) == 2
    assert all(item.track_id is not None for item in bottle_facts)
    person_appeared = next(event for event in person_events if event.event_type.value == "OBJECT_APPEARED")
    person_evidence = result.evidence.by_event(person_appeared)
    assert person_evidence[0].source_frame_index == person_appeared.frame_index == 0
    assert person_evidence[0].timestamp == person_appeared.timestamp == 0.0
    assert person_appeared.confidence == 0.9

    queried_person = result.queries.objects_appeared(label="person")
    assert len(queried_person) == 1 and queried_person[0].track_id == person_appeared.track_id
    history = result.queries.object_history(person_appeared.track_id)
    assert [event.timestamp for event in history] == [0.0, 0.1, 0.2, 0.5, 0.6]
    assert result.input_metadata.path == "mock_video.mp4"
    assert result.input_metadata.source_id == "video:mock_video.mp4"
    assert result.input_metadata.fps == 10.0
    assert result.input_metadata.source_sampled_frame_count == 7
    print("END-TO-END TRACKING, FACTS, EVIDENCE, AND QUERY ACCESS: PASS")


def test_one_object_video() -> None:
    pipeline = PerceptionPipeline(
        MockDetector({0: [("person", (14, 10, 26, 34))]}),
        depth_provider=None,
    )
    result = VideoAnalysisSession(
        MockVideoSource([make_frame(0)]),
        pipeline,
    ).run()
    assert result.status == "completed"
    assert len(result.timeline.events) == 1
    assert result.timeline.events[0].event_type.value == "OBJECT_APPEARED"
    assert len(result.facts.objects) == 1
    assert len(result.evidence.events) == 1
    assert len(result.episodes) == 1
    print("ONE-OBJECT END-TO-END VIDEO: PASS")


def test_invalid_frame_and_partial_pipeline_failure() -> None:
    frames = [
        make_frame(0, np.zeros((48, 100, 3), dtype=np.uint8)),
        make_frame(1),
    ]
    result = VideoAnalysisSession(
        MockVideoSource(frames),
        make_pipeline(),
    ).run()
    assert result.status == "partial"
    assert result.sampled_frame_count == result.processed_frame_count == 2
    assert result.invalid_perception_frame_count == 1
    assert result.errors
    assert result.timeline.events
    assert all(event.frame_index == 1 for event in result.timeline.events)
    print("INVALID PERCEPTION FRAME AND PARTIAL RESULT: PASS")


def test_result_serialization_keeps_structured_data() -> None:
    result = VideoAnalysisSession(
        MockVideoSource([make_frame(0)]),
        make_pipeline(),
    ).run()
    serialized = result.to_dict()
    assert serialized["status"] == "completed"
    assert serialized["sampled_frame_count"] == 1
    assert serialized["timeline"][0]["frame_index"] == 0
    assert serialized["facts"]["events"][0]["track_id"] is not None
    assert serialized["evidence"]["events"][0]["source_frame_index"] == 0
    assert isinstance(serialized["episodes"], list)
    print("STRUCTURED ANALYSIS RESULT SERIALIZATION: PASS")


def main() -> None:
    test_empty_and_unreadable_sources()
    test_one_object_video()
    test_end_to_end_tracking_facts_evidence_and_queries()
    test_invalid_frame_and_partial_pipeline_failure()
    test_result_serialization_keeps_structured_data()
    print("VIDEO ANALYSIS SESSION: PASS")


if __name__ == "__main__":
    main()
