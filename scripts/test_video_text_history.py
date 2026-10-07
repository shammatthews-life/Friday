from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.video.ocr import VideoTextObservation
from src.perception.video.text_history import (
    VideoTextEventType,
    VideoTextHistory,
    VideoTextQueryEngine,
)


def observation(
    text: str,
    timestamp: float,
    frame_index: int | None,
    *,
    source_id: str | None = "video:a",
    confidence: float | None = 0.9,
    bbox: tuple[float, float, float, float] | None = (1.0, 2.0, 30.0, 14.0),
) -> VideoTextObservation:
    return VideoTextObservation(
        text=text,
        confidence=confidence,
        timestamp=timestamp,
        frame_index=frame_index,
        bbox=bbox,
        source_id=source_id,
    )


def add_frame(
    history: VideoTextHistory,
    timestamp: float,
    frame_index: int | None,
    observations: tuple[VideoTextObservation, ...] = (),
    *,
    source_id: str | None = "video:a",
):
    return history.add_frame(
        timestamp=timestamp,
        frame_index=frame_index,
        source_id=source_id,
        observations=observations,
    )


def test_single_and_repeated_text_preserves_metadata() -> None:
    history = VideoTextHistory()
    added = add_frame(history, 1.0, 10, (observation("Room 204", 1.0, 10),))
    repeated = add_frame(
        history,
        1.1,
        11,
        (observation(" ROOM   204 ", 1.1, 11, confidence=0.8),),
    )
    assert [event.event_type for event in added] == [VideoTextEventType.APPEARED]
    assert [event.event_type for event in repeated] == [VideoTextEventType.REMAINED]
    episode = history.texts_present()[0]
    assert episode.normalized_text == "room 204"
    assert episode.text == "Room 204"
    assert episode.first_timestamp == 1.0 and episode.last_timestamp == 1.1
    assert episode.frame_indexes == (10, 11)
    assert episode.confidences == (0.9, 0.8)
    assert len(episode.bounding_boxes) == 2
    print("SINGLE/REPEATED TEXT GROUPING AND OBSERVED METADATA: PASS")


def test_disappearance_and_reappearance() -> None:
    history = VideoTextHistory(max_time_gap_seconds=0.5)
    add_frame(history, 0.0, 0, (observation("EXIT", 0.0, 0),))
    disappeared = add_frame(history, 0.8, 8)
    reappeared = add_frame(
        history,
        0.9,
        9,
        (observation("EXIT", 0.9, 9),),
    )
    assert [event.event_type for event in disappeared] == [VideoTextEventType.DISAPPEARED]
    assert disappeared[0].timestamp == 0.8 and disappeared[0].frame_index == 8
    assert disappeared[0].confidence is None and disappeared[0].bbox is None
    assert [event.event_type for event in reappeared] == [VideoTextEventType.REAPPEARED]
    assert len(history.texts_present()) == 1
    assert len(history.texts_disappeared()) == 1
    print("TEXT DISAPPEARANCE AND REAPPEARANCE: PASS")


def test_distinct_texts_and_sources_remain_separate() -> None:
    history = VideoTextHistory()
    add_frame(
        history,
        0.0,
        0,
        (
            observation("EXIT", 0.0, 0),
            observation("ROOM 204", 0.0, 0, bbox=(500.0, 400.0, 700.0, 450.0)),
        ),
    )
    add_frame(
        history,
        0.1,
        0,
        (observation("EXIT", 0.1, 0, source_id="video:b"),),
        source_id="video:b",
    )
    assert len(history.texts_present()) == 3
    assert len(history.texts_present(source_id="video:a")) == 2
    assert len(history.texts_present(source_id="video:b")) == 1
    assert len(history.text_history("exit")) == 2
    print("DIFFERENT TEXT AND SOURCE IDENTITIES REMAIN DISTINCT: PASS")


def test_missing_metadata_and_structured_queries() -> None:
    history = VideoTextHistory()
    no_metadata = observation(
        "NOTICE",
        2.0,
        None,
        source_id=None,
        confidence=None,
        bbox=None,
    )
    add_frame(history, 2.0, None, (no_metadata,), source_id=None)
    add_frame(history, 2.1, None, (observation("NOTICE", 2.1, None, source_id=None),), source_id=None)
    queries = VideoTextQueryEngine(history)
    current = queries.texts_present()
    assert len(current) == 1
    assert current[0].frame_indexes == ()
    assert current[0].confidences == (0.9,)
    assert current[0].bounding_boxes == ((1.0, 2.0, 30.0, 14.0),)
    events = queries.texts_in_time_range(2.05, 2.1, source_id=None)
    assert [event.timestamp for event in events] == [2.1]
    assert len(queries.texts_appeared()) == 1
    assert [event.event_type for event in queries.text_history(" notice ")] == [
        VideoTextEventType.APPEARED,
        VideoTextEventType.REMAINED,
    ]
    assert queries.texts_in_time_range(3.0, 4.0) == ()
    print("OPTIONAL METADATA AND STRUCTURED TEXT QUERIES: PASS")


def test_bounded_history_and_chronological_results() -> None:
    history = VideoTextHistory(history_size=2)
    add_frame(
        history,
        0.3,
        3,
        (observation("THIRD", 0.3, 3, source_id="video:c"),),
        source_id="video:c",
    )
    add_frame(
        history,
        0.1,
        1,
        (observation("FIRST", 0.1, 1, source_id="video:b"),),
        source_id="video:b",
    )
    add_frame(
        history,
        0.2,
        2,
        (observation("SECOND", 0.2, 2, source_id="video:d"),),
        source_id="video:d",
    )
    assert [event.timestamp for event in history.events] == [0.1, 0.2]
    assert len(history.episodes) == 2
    assert history.to_dict()["events"][0]["normalized_text"] == "first"
    print("BOUNDED HISTORY AND CHRONOLOGICAL EVENT ORDER: PASS")


def test_deduplicated_extractor_observations_feed_history() -> None:
    from src.core.scene_state import SceneState
    from src.perception.pipeline import PipelineOutput, PipelineTiming
    from src.perception.types import PerceptionFrame
    from src.perception.video.ocr import OCRDetection, VideoTextExtractor
    from src.perception.types import SceneSnapshot
    from src.perception.video.session import VideoAnalysisSession

    class Backend:
        def extract(self, image: np.ndarray) -> list[OCRDetection]:
            return [OCRDetection("CAUTION", 0.95, (2.0, 3.0, 40.0, 16.0))]

    class Source:
        def __init__(self, frames: list[PerceptionFrame]) -> None:
            self.frames = frames

        def __iter__(self):
            return iter(self.frames)

        def close(self) -> None:
            pass

    class Pipeline:
        def process(self, frame: PerceptionFrame) -> PipelineOutput:
            snapshot = SceneSnapshot(
                scene_state=SceneState(timestamp=frame.timestamp),
                timestamp=frame.timestamp,
                source_id=frame.source_id,
            )
            return PipelineOutput(snapshot=snapshot, timing=PipelineTiming())

    extractor = VideoTextExtractor(Backend())
    frames = []
    for index in range(2):
        frames.append(
            PerceptionFrame(
                image=np.zeros((4, 4, 3), dtype=np.uint8),
                timestamp=index / 10.0,
                source_id="video:extractor",
                frame_index=index,
            )
        )
    result = VideoAnalysisSession(
        Source(frames),
        Pipeline(),
        text_extractor=extractor,
    ).run()
    assert [event.event_type for event in result.text_history.events] == [
        VideoTextEventType.APPEARED,
        VideoTextEventType.REMAINED,
    ]
    assert len(result.text_observations) == 1
    assert result.text_queries.texts_present()[0].frame_indexes == (0, 1)
    assert result.to_dict()["text_history"]["events"][1]["frame_index"] == 1
    print("SESSION USES RAW DEDUPLICATED OCR OBSERVATIONS FOR TEXT HISTORY: PASS")


def main() -> None:
    test_single_and_repeated_text_preserves_metadata()
    test_disappearance_and_reappearance()
    test_distinct_texts_and_sources_remain_separate()
    test_missing_metadata_and_structured_queries()
    test_bounded_history_and_chronological_results()
    test_deduplicated_extractor_observations_feed_history()
    print("VIDEO TEXT HISTORY: PASS")


if __name__ == "__main__":
    main()
