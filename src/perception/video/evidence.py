from __future__ import annotations

from dataclasses import dataclass

from src.perception.video.facts import VideoFacts
from src.perception.video.timeline import SemanticEvent, SemanticEventType


@dataclass(frozen=True)
class VideoEventEvidence:
    event_type: SemanticEventType
    track_id: int | None
    label: str
    source_frame_index: int | None
    timestamp: float
    previous_frame_index: int | None
    previous_timestamp: float | None
    current_frame_index: int | None
    current_timestamp: float | None
    previous_position: tuple[float, float] | None
    current_position: tuple[float, float] | None
    confidence: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type.value,
            "track_id": self.track_id,
            "label": self.label,
            "source_frame_index": self.source_frame_index,
            "timestamp": self.timestamp,
            "previous_frame_index": self.previous_frame_index,
            "previous_timestamp": self.previous_timestamp,
            "current_frame_index": self.current_frame_index,
            "current_timestamp": self.current_timestamp,
            "previous_position": list(self.previous_position) if self.previous_position is not None else None,
            "current_position": list(self.current_position) if self.current_position is not None else None,
            "confidence": self.confidence,
        }


class VideoEvidenceIndex:
    """Bounded lookup index over evidence carried by semantic timeline events."""

    def __init__(self, history_size: int = 256) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        self.history_size = history_size
        self._evidence: list[VideoEventEvidence] = []

    @classmethod
    def from_facts(
        cls,
        facts: VideoFacts,
        *,
        history_size: int = 256,
    ) -> VideoEvidenceIndex:
        index = cls(history_size)
        previous_by_track: dict[int, SemanticEvent] = {}
        for event in facts.events:
            previous = previous_by_track.get(event.track_id) if event.track_id is not None else None
            index._append(_to_evidence(event, previous))
            if event.track_id is not None:
                previous_by_track[event.track_id] = event
        return index

    @property
    def events(self) -> tuple[VideoEventEvidence, ...]:
        return tuple(self._evidence)

    def by_event(self, event: SemanticEvent) -> tuple[VideoEventEvidence, ...]:
        return tuple(
            evidence
            for evidence in self._evidence
            if _matches_event(evidence, event)
        )

    def by_track_id(self, track_id: int) -> tuple[VideoEventEvidence, ...]:
        return tuple(item for item in self._evidence if item.track_id == track_id)

    def by_frame_index(self, frame_index: int) -> tuple[VideoEventEvidence, ...]:
        return tuple(
            item for item in self._evidence
            if item.source_frame_index == frame_index
        )

    def to_dict(self) -> dict[str, object]:
        return {"events": [item.to_dict() for item in self._evidence]}

    def _append(self, evidence: VideoEventEvidence) -> None:
        self._evidence.append(evidence)
        self._evidence.sort(key=lambda item: item.timestamp)
        self._evidence = self._evidence[-self.history_size :]


def _to_evidence(
    event: SemanticEvent,
    previous: SemanticEvent | None,
) -> VideoEventEvidence:
    return VideoEventEvidence(
        event_type=event.event_type,
        track_id=event.track_id,
        label=event.label,
        source_frame_index=event.frame_index,
        timestamp=event.timestamp,
        previous_frame_index=previous.frame_index if previous is not None else None,
        previous_timestamp=previous.timestamp if previous is not None else None,
        current_frame_index=event.frame_index,
        current_timestamp=event.timestamp,
        previous_position=event.previous_position,
        current_position=event.current_position,
        confidence=event.confidence,
    )


def _matches_event(evidence: VideoEventEvidence, event: SemanticEvent) -> bool:
    return (
        evidence.event_type is event.event_type
        and evidence.track_id == event.track_id
        and evidence.label == event.label
        and evidence.source_frame_index == event.frame_index
        and evidence.timestamp == event.timestamp
        and evidence.previous_position == event.previous_position
        and evidence.current_position == event.current_position
        and evidence.confidence == event.confidence
    )
