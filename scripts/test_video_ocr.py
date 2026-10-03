from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.scene_state import SceneState
from src.perception.pipeline import PipelineOutput, PipelineTiming
from src.perception.types import PerceptionFrame, SceneSnapshot
from src.perception.video.ocr import OCRDetection, VideoTextExtractor
from src.perception.video.session import VideoAnalysisSession


class MockOCRBackend:
    def __init__(self, results: dict[int, list[OCRDetection]]) -> None:
        self.results = results
        self.calls = 0

    def extract(self, image: np.ndarray) -> list[OCRDetection]:
        index = int(image[0, 0, 0])
        self.calls += 1
        return self.results.get(index, [])


class MockSource:
    def __init__(self, frames: list[PerceptionFrame]) -> None:
        self.frames = frames
        self.closed = False

    def __iter__(self):
        return iter(self.frames)

    def close(self) -> None:
        self.closed = True


class MockPipeline:
    def process(self, frame: PerceptionFrame) -> PipelineOutput:
        snapshot = SceneSnapshot(
            scene_state=SceneState(timestamp=frame.timestamp),
            timestamp=frame.timestamp,
            source_id=frame.source_id,
        )
        return PipelineOutput(snapshot=snapshot, timing=PipelineTiming())


def make_frame(
    index: int,
    *,
    timestamp: float | None = None,
    source_id: str = "video:ocr-test",
    frame_index: int | None = -1,
) -> PerceptionFrame:
    image = np.full((4, 4, 3), index, dtype=np.uint8)
    return PerceptionFrame(
        image=image,
        timestamp=timestamp if timestamp is not None else index / 10.0,
        source_id=source_id,
        frame_index=index if frame_index == -1 else frame_index,
    )


def test_single_and_multiple_text_observations() -> None:
    backend = MockOCRBackend(
        {
            1: [
                OCRDetection("ROOM 204", 0.91, (100, 80, 300, 140)),
                OCRDetection("EXIT", 0.87, (15, 20, 50, 35)),
            ]
        }
    )
    extractor = VideoTextExtractor(backend)
    observations = extractor.process(make_frame(1, timestamp=4.2, frame_index=126))
    assert [item.text for item in observations] == ["ROOM 204", "EXIT"]
    first = observations[0]
    assert first.timestamp == 4.2
    assert first.frame_index == 126
    assert first.confidence == 0.91
    assert first.bbox == (100, 80, 300, 140)
    assert first.source_id == "video:ocr-test"
    print("SINGLE AND MULTIPLE STRUCTURED OCR OBSERVATIONS: PASS")


def test_temporal_deduplication_and_distinct_text() -> None:
    backend = MockOCRBackend(
        {
            0: [OCRDetection("ROOM 204", 0.9)],
            1: [OCRDetection(" room   204 ", 0.92)],
            2: [OCRDetection("EXIT", 0.88)],
            3: [OCRDetection("ROOM 204", 0.9)],
        }
    )
    extractor = VideoTextExtractor(
        backend,
        dedup_frame_gap=1,
        dedup_time_gap_seconds=1.0,
    )
    assert [item.text for item in extractor.process(make_frame(0))] == ["ROOM 204"]
    assert extractor.process(make_frame(1)) == ()
    assert [item.text for item in extractor.process(make_frame(2))] == ["EXIT"]
    assert [item.text for item in extractor.process(make_frame(3))] == ["ROOM 204"]
    assert extractor.duplicates_suppressed == 1
    assert [item.text for item in extractor.observations] == ["ROOM 204", "EXIT", "ROOM 204"]
    print("NEARBY REPEATED TEXT SUPPRESSION AND DIFFERENT TEXT: PASS")


def test_optional_metadata_and_empty_results() -> None:
    backend = MockOCRBackend({4: [OCRDetection("NOTICE")]})
    extractor = VideoTextExtractor(backend)
    frame = make_frame(4, timestamp=2.5, source_id="", frame_index=4)
    observations = extractor.process(frame)
    assert len(observations) == 1
    assert observations[0].timestamp == 2.5
    assert observations[0].frame_index == 4
    assert observations[0].confidence is None
    assert observations[0].bbox is None
    assert observations[0].source_id is None

    missing_frame_metadata = make_frame(5, source_id="", frame_index=None)
    no_frame_index_backend = MockOCRBackend({5: [OCRDetection("WARNING", None, None)]})
    without_index = VideoTextExtractor(no_frame_index_backend).process(missing_frame_metadata)[0]
    assert without_index.frame_index is None
    assert without_index.source_id is None
    assert without_index.confidence is None and without_index.bbox is None

    assert extractor.process(make_frame(6)) == ()
    assert backend.calls == 2
    print("OPTIONAL METADATA, MISSING FRAME INDEX, AND EMPTY OCR RESULT: PASS")


def test_dedup_uses_time_when_frame_index_is_missing() -> None:
    backend = MockOCRBackend({1: [OCRDetection("CAUTION")], 2: [OCRDetection("CAUTION")]})
    extractor = VideoTextExtractor(backend, dedup_time_gap_seconds=1.0)
    first = make_frame(1, timestamp=10.0, frame_index=None)
    second = make_frame(2, timestamp=10.5, frame_index=None)
    assert len(extractor.process(first)) == 1
    assert extractor.process(second) == ()
    assert extractor.duplicates_suppressed == 1
    print("TIMESTAMP-BASED DEDUP WITH ABSENT FRAME INDEX: PASS")


def test_bounded_history_and_session_data_flow() -> None:
    backend = MockOCRBackend(
        {
            index: [OCRDetection(f"TEXT {index}", confidence=None)]
            for index in range(3)
        }
    )
    extractor = VideoTextExtractor(backend, history_size=2)
    source = MockSource([make_frame(index) for index in range(3)])
    result = VideoAnalysisSession(
        source,
        MockPipeline(),
        text_extractor=extractor,
    ).run()
    assert source.closed
    assert len(result.text_observations) == 2
    assert [item.text for item in result.text_observations] == ["TEXT 1", "TEXT 2"]
    assert result.to_dict()["text_observations"][0]["frame_index"] == 1
    print("BOUNDED OCR HISTORY AND SESSION RESULT INTEGRATION: PASS")


def main() -> None:
    test_single_and_multiple_text_observations()
    test_temporal_deduplication_and_distinct_text()
    test_optional_metadata_and_empty_results()
    test_dedup_uses_time_when_frame_index_is_missing()
    test_bounded_history_and_session_data_flow()
    print("VIDEO OCR FOUNDATION: PASS")


if __name__ == "__main__":
    main()
