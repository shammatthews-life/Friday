from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.perception.pipeline import PerceptionPipeline
from src.perception.types import PerceptionFrame
from src.perception.video.evidence import VideoEvidenceIndex
from src.perception.video.facts import VideoFacts
from src.perception.video.ocr import OCRBackendError, VideoTextExtractor, VideoTextObservation
from src.perception.video.query import VideoQueryEngine
from src.perception.video.summary import VideoEpisode, VideoEventSummarizer
from src.perception.video.temporal import TemporalSampler
from src.perception.video.timeline import SemanticEvent, SemanticTimeline
from src.perception.video.video_source import VideoInputError


@dataclass(frozen=True)
class VideoInputMetadata:
    path: str | None
    source_id: str | None
    fps: float | None
    decoded_frame_count: int | None
    source_sampled_frame_count: int | None
    source_skipped_invalid_frame_count: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "source_id": self.source_id,
            "fps": self.fps,
            "decoded_frame_count": self.decoded_frame_count,
            "source_sampled_frame_count": self.source_sampled_frame_count,
            "source_skipped_invalid_frame_count": self.source_skipped_invalid_frame_count,
        }


@dataclass(frozen=True)
class VideoAnalysisResult:
    status: str
    input_metadata: VideoInputMetadata
    sampled_frame_count: int
    processed_frame_count: int
    invalid_perception_frame_count: int
    timeline: SemanticTimeline
    facts: VideoFacts
    evidence: VideoEvidenceIndex
    episodes: tuple[VideoEpisode, ...]
    queries: VideoQueryEngine
    errors: tuple[str, ...]
    text_observations: tuple[VideoTextObservation, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "input_metadata": self.input_metadata.to_dict(),
            "sampled_frame_count": self.sampled_frame_count,
            "processed_frame_count": self.processed_frame_count,
            "invalid_perception_frame_count": self.invalid_perception_frame_count,
            "timeline": [_event_to_dict(event) for event in self.timeline.events],
            "facts": self.facts.to_dict(),
            "evidence": self.evidence.to_dict(),
            "episodes": [episode.to_dict() for episode in self.episodes],
            "text_observations": [item.to_dict() for item in self.text_observations],
            "errors": list(self.errors),
        }


class VideoAnalysisSession:
    """Run sampled frames through the existing local video perception components."""

    def __init__(
        self,
        source: Iterable[PerceptionFrame],
        pipeline: PerceptionPipeline,
        *,
        temporal_sampler: TemporalSampler | None = None,
        timeline: SemanticTimeline | None = None,
        text_extractor: VideoTextExtractor | None = None,
        history_size: int = 256,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        self.source = source
        self.pipeline = pipeline
        self.temporal_sampler = temporal_sampler or TemporalSampler()
        self.timeline = timeline or SemanticTimeline(history_size=history_size)
        self.text_extractor = text_extractor
        self.history_size = history_size

    def run(self) -> VideoAnalysisResult:
        sampled_count = 0
        processed_count = 0
        invalid_count = 0
        errors: list[str] = []
        source_error: VideoInputError | None = None
        first_source_id: str | None = None

        try:
            iterator = iter(self.source)
        except VideoInputError as error:
            iterator = iter(())
            source_error = error
            errors.append(f"video input failed: {error}")

        try:
            for frame in iterator:
                sampled_count += 1
                if first_source_id is None:
                    first_source_id = frame.source_id or None
                self.temporal_sampler.process(frame)
                if self.text_extractor is not None:
                    try:
                        self.text_extractor.process(frame)
                    except OCRBackendError as error:
                        errors.append(
                            f"frame {frame.frame_index}: OCR failed: {type(error).__name__}: {error}"
                        )
                output = self.pipeline.process(frame)
                processed_count += 1
                if not output.snapshot.valid:
                    invalid_count += 1
                    if output.snapshot.error:
                        errors.append(
                            f"frame {frame.frame_index}: {output.snapshot.error}"
                        )
                    continue
                self.timeline.add_observation(
                    output.snapshot,
                    frame_index=frame.frame_index,
                )
        except VideoInputError as error:
            source_error = error
            errors.append(f"video input failed: {error}")
        finally:
            close = getattr(self.source, "close", None)
            if callable(close):
                close()

        decoder_error = getattr(self.source, "last_error", None)
        if decoder_error:
            errors.append(f"video decoding failed: {decoder_error}")
        facts = VideoFacts.from_timeline(self.timeline, history_size=self.history_size)
        evidence = VideoEvidenceIndex.from_facts(facts, history_size=self.history_size)
        episodes = VideoEventSummarizer(history_size=self.history_size).summarize(
            facts,
            evidence,
        )
        queries = VideoQueryEngine(
            self.timeline,
            facts=facts,
            evidence_index=evidence,
            episodes=episodes,
        )
        metadata = _input_metadata(self.source, first_source_id)
        if metadata.source_skipped_invalid_frame_count:
            errors.append(
                f"video source skipped {metadata.source_skipped_invalid_frame_count} invalid frame(s)"
            )

        if source_error is not None and sampled_count == 0:
            status = "unreadable"
        elif sampled_count == 0 and not decoder_error and not metadata.source_skipped_invalid_frame_count:
            status = "empty"
        elif errors or invalid_count:
            status = "partial"
        else:
            status = "completed"

        return VideoAnalysisResult(
            status=status,
            input_metadata=metadata,
            sampled_frame_count=sampled_count,
            processed_frame_count=processed_count,
            invalid_perception_frame_count=invalid_count,
            timeline=self.timeline,
            facts=facts,
            evidence=evidence,
            episodes=episodes,
            queries=queries,
            errors=tuple(errors),
            text_observations=(
                self.text_extractor.observations
                if self.text_extractor is not None
                else ()
            ),
        )


def _input_metadata(
    source: Any,
    source_id: str | None,
) -> VideoInputMetadata:
    video_path = getattr(source, "video_path", None)
    path = str(video_path) if isinstance(video_path, (str, Path)) else None
    return VideoInputMetadata(
        path=path,
        source_id=source_id,
        fps=_optional_number(getattr(source, "fps", None)),
        decoded_frame_count=_optional_integer(getattr(source, "decoded_frames", None)),
        source_sampled_frame_count=_optional_integer(getattr(source, "yielded_frames", None)),
        source_skipped_invalid_frame_count=_optional_integer(
            getattr(source, "skipped_invalid_frames", None)
        ),
    )


def _optional_number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _optional_integer(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def _event_to_dict(event: SemanticEvent) -> dict[str, object]:
    return {
        "event_type": event.event_type.value,
        "timestamp": event.timestamp,
        "frame_index": event.frame_index,
        "track_id": event.track_id,
        "label": event.label,
        "previous_position": list(event.previous_position) if event.previous_position is not None else None,
        "current_position": list(event.current_position) if event.current_position is not None else None,
        "confidence": event.confidence,
    }
