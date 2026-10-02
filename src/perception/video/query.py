from __future__ import annotations

from dataclasses import dataclass

from src.perception.video.evidence import VideoEventEvidence, VideoEvidenceIndex
from src.perception.video.facts import ObjectVideoFacts, VideoFacts
from src.perception.video.summary import VideoEpisode, VideoEventSummarizer
from src.perception.video.timeline import SemanticEvent, SemanticEventType, SemanticTimeline


@dataclass(frozen=True)
class ObjectPresenceDuration:
    track_id: int | None
    label: str
    duration_seconds: float
    first_seen_timestamp: float | None
    last_seen_timestamp: float | None
    is_present: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "duration_seconds": self.duration_seconds,
            "first_seen_timestamp": self.first_seen_timestamp,
            "last_seen_timestamp": self.last_seen_timestamp,
            "is_present": self.is_present,
        }


class VideoQueryEngine:
    """Deterministic structured queries over retained video facts and evidence."""

    def __init__(
        self,
        timeline: SemanticTimeline,
        *,
        facts: VideoFacts | None = None,
        evidence_index: VideoEvidenceIndex | None = None,
        episodes: tuple[VideoEpisode, ...] | None = None,
    ) -> None:
        self.timeline_source = timeline
        self.facts = facts or VideoFacts.from_timeline(timeline)
        self.evidence_index = evidence_index or VideoEvidenceIndex.from_facts(self.facts)
        self.episodes = (
            episodes
            if episodes is not None
            else VideoEventSummarizer().summarize(self.facts, self.evidence_index)
        )

    def objects_present(self, *, label: str | None = None) -> tuple[ObjectVideoFacts, ...]:
        return self._filter_objects(self.facts.objects_present, label=label)

    def objects_appeared(self, *, label: str | None = None) -> tuple[ObjectVideoFacts, ...]:
        return self._filter_objects(self.facts.objects_appeared, label=label)

    def objects_disappeared(self, *, label: str | None = None) -> tuple[ObjectVideoFacts, ...]:
        return self._filter_objects(self.facts.objects_disappeared, label=label)

    def objects_moved(
        self,
        *,
        track_id: int | None = None,
        label: str | None = None,
    ) -> tuple[SemanticEvent, ...]:
        return self._filter_events(
            self.facts.movement_events,
            track_id=track_id,
            label=label,
        )

    def object_history(self, track_id: int) -> tuple[SemanticEvent, ...]:
        return self.facts.object_history(track_id)

    def object_presence_duration(self, track_id: int) -> ObjectPresenceDuration | None:
        item = next((obj for obj in self.facts.objects if obj.track_id == track_id), None)
        if item is None:
            return None
        return ObjectPresenceDuration(
            track_id=item.track_id,
            label=item.label,
            duration_seconds=item.presence_duration_seconds,
            first_seen_timestamp=item.first_seen_timestamp,
            last_seen_timestamp=item.last_seen_timestamp,
            is_present=item.is_present,
        )

    def events_in_time_range(
        self,
        start_timestamp: float,
        end_timestamp: float,
        *,
        track_id: int | None = None,
        label: str | None = None,
    ) -> tuple[SemanticEvent, ...]:
        self._validate_range(start_timestamp, end_timestamp)
        matching = (
            event for event in self.facts.events
            if start_timestamp <= event.timestamp <= end_timestamp
        )
        return self._filter_events(matching, track_id=track_id, label=label)

    def episodes_in_time_range(
        self,
        start_timestamp: float,
        end_timestamp: float,
        *,
        track_id: int | None = None,
        label: str | None = None,
    ) -> tuple[VideoEpisode, ...]:
        self._validate_range(start_timestamp, end_timestamp)
        return tuple(
            episode
            for episode in self.episodes
            if (
                (
                    episode.start_timestamp <= end_timestamp
                    and episode.end_timestamp >= start_timestamp
                )
                if episode.end_timestamp is not None
                else any(
                    start_timestamp <= event.timestamp <= end_timestamp
                    for event in episode.events
                )
            )
            and (track_id is None or episode.track_id == track_id)
            and (label is None or episode.label.casefold() == label.casefold())
        )

    def evidence_for_event(self, event: SemanticEvent) -> tuple[VideoEventEvidence, ...]:
        return self.evidence_index.by_event(event)

    def timeline(self) -> tuple[SemanticEvent, ...]:
        return self.facts.event_timeline

    @staticmethod
    def _validate_range(start_timestamp: float, end_timestamp: float) -> None:
        if start_timestamp > end_timestamp:
            raise ValueError("start_timestamp must not be after end_timestamp")

    def _filter_objects(
        self,
        objects: tuple[ObjectVideoFacts, ...],
        *,
        label: str | None,
    ) -> tuple[ObjectVideoFacts, ...]:
        if label is None:
            return objects
        normalized = label.casefold()
        return tuple(item for item in objects if item.label.casefold() == normalized)

    @staticmethod
    def _filter_events(
        events,
        *,
        track_id: int | None,
        label: str | None,
    ) -> tuple[SemanticEvent, ...]:
        normalized = label.casefold() if label is not None else None
        return tuple(
            event for event in events
            if (track_id is None or event.track_id == track_id)
            and (normalized is None or event.label.casefold() == normalized)
        )
