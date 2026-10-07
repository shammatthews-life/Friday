from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from src.perception.video.evidence import VideoEvidenceIndex, VideoEventEvidence
from src.perception.video.facts import ObjectVideoFacts, VideoFacts
from src.perception.video.ocr import VideoTextObservation
from src.perception.video.summary import VideoEpisode
from src.perception.video.text_history import VideoTextHistory
from src.perception.video.timeline import SemanticEvent

if TYPE_CHECKING:
    from src.perception.video.session import VideoAnalysisResult


@dataclass(frozen=True)
class VideoKnowledge:
    """Structured view over existing visual, evidence, and OCR analysis layers."""

    facts: VideoFacts
    episodes: tuple[VideoEpisode, ...]
    evidence_index: VideoEvidenceIndex
    text_history: VideoTextHistory
    text_observations: tuple[VideoTextObservation, ...] = ()
    video_metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_analysis(cls, result: VideoAnalysisResult) -> VideoKnowledge:
        metadata = {
            "status": result.status,
            "input": result.input_metadata.to_dict(),
            "sampled_frame_count": result.sampled_frame_count,
            "processed_frame_count": result.processed_frame_count,
            "invalid_perception_frame_count": result.invalid_perception_frame_count,
        }
        return cls(
            facts=result.facts,
            episodes=result.episodes,
            evidence_index=result.evidence,
            text_history=result.text_history,
            text_observations=result.text_observations,
            video_metadata=metadata,
        )

    def all_visual_information(self) -> dict[str, object]:
        return {
            "objects": [item.to_dict() for item in self.facts.objects],
            "events": [
                _event_to_dict(event)
                for _, event in sorted(
                    enumerate(self.facts.events),
                    key=lambda item: (item[1].timestamp, item[0]),
                )
            ],
            "episodes": [
                episode.to_dict()
                for _, episode in sorted(
                    enumerate(self.episodes),
                    key=lambda item: (item[1].start_timestamp, item[0]),
                )
            ],
            "evidence_references": self.evidence_index.to_dict()["events"],
        }

    def all_text_information(self) -> dict[str, object]:
        return {
            "observations": [
                item.to_dict()
                for _, item in sorted(
                    enumerate(self.text_observations),
                    key=lambda item: (item[1].timestamp, item[0]),
                )
            ],
            "history": self.text_history.to_dict(),
        }

    def events_in_time_range(
        self,
        start_timestamp: float,
        end_timestamp: float,
    ) -> tuple[SemanticEvent, ...]:
        _validate_range(start_timestamp, end_timestamp)
        return tuple(
            event
            for _, event in sorted(
                (
                    (index, event)
                    for index, event in enumerate(self.facts.events)
                    if start_timestamp <= event.timestamp <= end_timestamp
                ),
                key=lambda item: (item[1].timestamp, item[0]),
            )
        )

    def object_information(self, track_id: int) -> ObjectVideoFacts | None:
        return next(
            (item for item in self.facts.objects if item.track_id == track_id),
            None,
        )

    def text_information_in_time_range(
        self,
        start_timestamp: float,
        end_timestamp: float,
        *,
        source_id: str | None = None,
    ) -> dict[str, object]:
        _validate_range(start_timestamp, end_timestamp)
        observations = tuple(
            observation
            for _, observation in sorted(
                (
                    (index, observation)
                    for index, observation in enumerate(self.text_observations)
                    if start_timestamp <= observation.timestamp <= end_timestamp
                    and (source_id is None or observation.source_id == source_id)
                ),
                key=lambda item: (item[1].timestamp, item[0]),
            )
        )
        events = self.text_history.texts_in_time_range(
            start_timestamp,
            end_timestamp,
            source_id=source_id,
        )
        return {
            "observations": [item.to_dict() for item in observations],
            "events": [item.to_dict() for item in events],
        }

    def evidence_references(
        self,
        *,
        track_id: int | None = None,
        frame_index: int | None = None,
    ) -> tuple[VideoEventEvidence, ...]:
        return tuple(
            item
            for item in self.evidence_index.events
            if (track_id is None or item.track_id == track_id)
            and (frame_index is None or item.source_frame_index == frame_index)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "visual": self.all_visual_information(),
            "text": self.all_text_information(),
            "video_metadata": dict(self.video_metadata),
        }


def _validate_range(start_timestamp: float, end_timestamp: float) -> None:
    if start_timestamp > end_timestamp:
        raise ValueError("start_timestamp must not be after end_timestamp")


def _event_to_dict(event: SemanticEvent) -> dict[str, object]:
    return {
        "event_type": event.event_type.value,
        "timestamp": event.timestamp,
        "frame_index": event.frame_index,
        "track_id": event.track_id,
        "label": event.label,
        "previous_position": (
            list(event.previous_position) if event.previous_position is not None else None
        ),
        "current_position": (
            list(event.current_position) if event.current_position is not None else None
        ),
        "confidence": event.confidence,
    }
