from __future__ import annotations

from dataclasses import dataclass

from src.perception.video.evidence import VideoEventEvidence, VideoEvidenceIndex
from src.perception.video.facts import VideoFacts
from src.perception.video.timeline import SemanticEvent, SemanticEventType


@dataclass(frozen=True)
class VideoEpisode:
    episode_type: str
    track_id: int | None
    label: str
    start_timestamp: float
    end_timestamp: float | None
    events: tuple[SemanticEvent, ...]
    positions: tuple[tuple[float, float], ...]
    confidences: tuple[float, ...]
    evidence: tuple[VideoEventEvidence, ...]
    incomplete: bool

    @property
    def event_types(self) -> tuple[SemanticEventType, ...]:
        return tuple(event.event_type for event in self.events)

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_type": self.episode_type,
            "track_id": self.track_id,
            "label": self.label,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "event_types": [event_type.value for event_type in self.event_types],
            "events": [_event_to_dict(event) for event in self.events],
            "positions": [list(position) for position in self.positions],
            "confidences": list(self.confidences),
            "evidence": [item.to_dict() for item in self.evidence],
            "incomplete": self.incomplete,
        }


class VideoEventSummarizer:
    """Group retained semantic events into bounded per-track episodes."""

    def __init__(self, history_size: int = 128) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        self.history_size = history_size

    def summarize(
        self,
        facts: VideoFacts,
        evidence_index: VideoEvidenceIndex | None = None,
    ) -> tuple[VideoEpisode, ...]:
        ordered = tuple(
            event
            for _, event in sorted(
                enumerate(facts.events),
                key=lambda item: (item[1].timestamp, item[0]),
            )
        )
        grouped: dict[tuple[str, int], list[SemanticEvent]] = {}
        for index, event in enumerate(ordered):
            identity = ("track", event.track_id) if event.track_id is not None else ("event", index)
            grouped.setdefault(identity, []).append(event)

        episodes: list[VideoEpisode] = []
        for track_events in grouped.values():
            episodes.extend(self._summarize_track(track_events, evidence_index))
        episodes.sort(key=lambda episode: episode.start_timestamp)
        return tuple(episodes[-self.history_size :])

    def _summarize_track(
        self,
        events: list[SemanticEvent],
        evidence_index: VideoEvidenceIndex | None,
    ) -> list[VideoEpisode]:
        episodes: list[VideoEpisode] = []
        current: list[SemanticEvent] = []
        waiting_for_reacquisition = False
        episode_type = "presence"
        starts_midstream = False

        def finish(*, end_timestamp: float | None, incomplete: bool) -> None:
            nonlocal current, waiting_for_reacquisition
            if current:
                episodes.append(
                    self._make_episode(
                        current,
                        episode_type=episode_type,
                        end_timestamp=end_timestamp,
                        incomplete=incomplete or starts_midstream,
                        evidence_index=evidence_index,
                    )
                )
            current = []
            waiting_for_reacquisition = False

        for event in events:
            event_type = event.event_type
            if event_type is SemanticEventType.OBJECT_APPEARED:
                if current:
                    previous_end = (
                        current[-1].timestamp
                        if current[-1].event_type is SemanticEventType.OBJECT_DISAPPEARED
                        else None
                    )
                    finish(end_timestamp=previous_end, incomplete=previous_end is None)
                current = [event]
                episode_type = "presence"
                starts_midstream = False
                continue

            if event_type is SemanticEventType.OBJECT_REACQUIRED:
                if waiting_for_reacquisition:
                    current.append(event)
                    episode_type = "interrupted_reappearance"
                    waiting_for_reacquisition = False
                elif current:
                    current.append(event)
                    episode_type = "interrupted_reappearance"
                else:
                    current = [event]
                    episode_type = "reappearance"
                    starts_midstream = True
                continue

            if event_type is SemanticEventType.OBJECT_DISAPPEARED:
                if not current:
                    current = [event]
                    episode_type = "disappearance"
                    starts_midstream = True
                else:
                    current.append(event)
                waiting_for_reacquisition = True
                continue

            if event_type in {SemanticEventType.OBJECT_REMAINED, SemanticEventType.OBJECT_MOVED}:
                if waiting_for_reacquisition:
                    finish(end_timestamp=current[-1].timestamp, incomplete=False)
                    current = [event]
                    episode_type = "activity"
                    starts_midstream = True
                elif current:
                    current.append(event)
                    if event_type is SemanticEventType.OBJECT_MOVED and episode_type == "presence":
                        episode_type = "activity"
                else:
                    current = [event]
                    episode_type = "activity"
                    starts_midstream = True

        if current:
            ended_on_disappearance = (
                current[-1].event_type is SemanticEventType.OBJECT_DISAPPEARED
            )
            finish(
                end_timestamp=current[-1].timestamp if ended_on_disappearance else None,
                incomplete=not ended_on_disappearance,
            )
        return episodes

    @staticmethod
    def _make_episode(
        events: list[SemanticEvent],
        *,
        episode_type: str,
        end_timestamp: float | None,
        incomplete: bool,
        evidence_index: VideoEvidenceIndex | None,
    ) -> VideoEpisode:
        positions: list[tuple[float, float]] = []
        confidences: list[float] = []
        evidence: list[VideoEventEvidence] = []
        for event in events:
            for position in (event.previous_position, event.current_position):
                if position is not None and (not positions or positions[-1] != position):
                    positions.append(position)
            if event.confidence is not None:
                confidences.append(event.confidence)
            if evidence_index is not None:
                evidence.extend(evidence_index.by_event(event))

        return VideoEpisode(
            episode_type=episode_type,
            track_id=events[0].track_id,
            label=events[0].label,
            start_timestamp=events[0].timestamp,
            end_timestamp=end_timestamp,
            events=tuple(events),
            positions=tuple(positions),
            confidences=tuple(confidences),
            evidence=tuple(evidence),
            incomplete=incomplete,
        )


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
