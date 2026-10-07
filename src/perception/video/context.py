from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from src.perception.video.facts import ObjectVideoFacts
from src.perception.video.knowledge import VideoKnowledge, semantic_event_to_dict
from src.perception.video.text_history import VideoTextEpisode, VideoTextEvent
from src.perception.video.timeline import SemanticEvent


@dataclass(frozen=True)
class VideoContextLimits:
    max_objects: int = 100
    max_events: int = 200
    max_episodes: int = 100
    max_text_observations: int = 200
    max_text_history_episodes: int = 100
    max_text_history_events: int = 200
    max_evidence_references: int = 200
    max_nested_events: int = 50
    max_nested_evidence: int = 50
    max_nested_text_observations: int = 50

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")


class VideoContextBuilder:
    """Build bounded, deterministic structured context from grounded knowledge."""

    def __init__(
        self,
        knowledge: VideoKnowledge,
        *,
        limits: VideoContextLimits | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.limits = limits or VideoContextLimits()
        self._truncated: dict[str, int] = {}

    def build(self) -> dict[str, object]:
        return self._build({"kind": "full_video"})

    def for_time_range(
        self,
        start_timestamp: float,
        end_timestamp: float,
    ) -> dict[str, object]:
        _validate_range(start_timestamp, end_timestamp)
        return self._build(
            {
                "kind": "time_range",
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
            },
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )

    def for_object(self, track_id: int) -> dict[str, object]:
        return self._build({"kind": "object", "track_id": track_id}, track_id=track_id)

    def for_label(self, label: str) -> dict[str, object]:
        normalized_label = " ".join(label.casefold().split())
        return self._build(
            {"kind": "object_label", "label": normalized_label},
            label=normalized_label,
        )

    def for_visual_events(self) -> dict[str, object]:
        return self._build({"kind": "visual_events"})

    def for_all_text(self) -> dict[str, object]:
        return self._build({"kind": "text_all"})

    def for_text(
        self,
        text: str,
        *,
        source_id: str | None = None,
    ) -> dict[str, object]:
        normalized_text = _normalize(text)
        return self._build(
            {
                "kind": "text",
                "normalized_text": normalized_text,
                "source_id": source_id,
            },
            normalized_text=normalized_text,
            source_id=source_id,
        )

    def _build(
        self,
        selection: dict[str, object],
        *,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
        track_id: int | None = None,
        label: str | None = None,
        normalized_text: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, object]:
        self._truncated = {}
        time_filtered = start_timestamp is not None
        object_filtered = selection["kind"] in {"object", "object_label"}
        text_filtered = selection["kind"] in {"text", "text_all"}
        visual_enabled = not text_filtered
        text_enabled = not object_filtered and selection["kind"] != "visual_events"

        events = [
            event
            for event in self.knowledge.facts.events
            if visual_enabled
            and (track_id is None or event.track_id == track_id)
            and (label is None or event.label.casefold() == label)
            and _in_range(event.timestamp, start_timestamp, end_timestamp)
        ]
        events = _ordered(events, lambda item: item.timestamp)

        objects = [
            item
            for item in self.knowledge.facts.objects
            if visual_enabled
            and (track_id is None or item.track_id == track_id)
            and (label is None or item.label.casefold() == label)
            and (
                not time_filtered
                or any(_in_range(event.timestamp, start_timestamp, end_timestamp) for event in item.events)
            )
        ]
        objects.sort(
            key=lambda item: (
                float("-inf") if item.first_seen_timestamp is None else item.first_seen_timestamp,
                -1 if item.track_id is None else item.track_id,
                item.label.casefold(),
            )
        )

        episodes = [
            item
            for item in self.knowledge.episodes
            if visual_enabled
            and (track_id is None or item.track_id == track_id)
            and (label is None or item.label.casefold() == label)
            and any(
                _in_range(event.timestamp, start_timestamp, end_timestamp)
                for event in item.events
            )
        ]
        episodes = _ordered(episodes, lambda item: item.start_timestamp)

        evidence = [
            item
            for item in self.knowledge.evidence_index.events
            if visual_enabled
            and (track_id is None or item.track_id == track_id)
            and (label is None or item.label.casefold() == label)
            and _in_range(item.timestamp, start_timestamp, end_timestamp)
        ]
        evidence = _ordered(evidence, lambda item: item.timestamp)

        observations = [
            item
            for item in self.knowledge.text_observations
            if text_enabled
            and (normalized_text is None or _normalize(item.text) == normalized_text)
            and (source_id is None or item.source_id == source_id)
            and _in_range(item.timestamp, start_timestamp, end_timestamp)
        ]
        observations = _ordered(observations, lambda item: item.timestamp)

        history_events = [
            item
            for item in self.knowledge.text_history.events
            if text_enabled
            and (normalized_text is None or item.normalized_text == normalized_text)
            and (source_id is None or item.source_id == source_id)
            and _in_range(item.timestamp, start_timestamp, end_timestamp)
        ]
        history_events = _ordered(history_events, lambda item: item.timestamp)
        history_episodes = [
            item
            for item in self.knowledge.text_history.episodes
            if text_enabled
            and (normalized_text is None or item.normalized_text == normalized_text)
            and (source_id is None or item.source_id == source_id)
            and (
                not time_filtered
                or any(
                    item.source_id == event.source_id
                    and item.normalized_text == event.normalized_text
                    and _event_in_text_episode(item, event)
                    for event in history_events
                )
            )
        ]
        history_episodes = _ordered(
            history_episodes,
            lambda item: item.first_timestamp,
        )

        limited_objects = self._limit(objects, "visual_objects", self.limits.max_objects)
        limited_events = self._limit(events, "visual_events", self.limits.max_events)
        limited_episodes = self._limit(episodes, "episodes", self.limits.max_episodes)
        limited_observations = self._limit(
            observations,
            "text_observations",
            self.limits.max_text_observations,
        )
        limited_history_episodes = self._limit(
            history_episodes,
            "text_history_episodes",
            self.limits.max_text_history_episodes,
        )
        limited_history_events = self._limit(
            history_events,
            "text_history_events",
            self.limits.max_text_history_events,
        )
        limited_evidence = self._limit(
            evidence,
            "evidence_references",
            self.limits.max_evidence_references,
        )

        event_dicts = [semantic_event_to_dict(item) for item in limited_events]
        text_observation_dicts = [item.to_dict() for item in limited_observations]
        text_history_event_dicts = [item.to_dict() for item in limited_history_events]
        text_history_episode_dicts = [
            self._text_episode_dict(
                item,
                time_filtered=time_filtered,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                selected_events=[
                    event
                    for event in history_events
                    if event.source_id == item.source_id
                    and event.normalized_text == item.normalized_text
                    and _event_in_text_episode(item, event)
                ],
            )
            for item in limited_history_episodes
        ]
        visual_objects = [
            self._object_dict(
                item,
                time_filtered=time_filtered,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
            for item in limited_objects
        ]
        visual_episodes = [
            self._episode_dict(item, start_timestamp, end_timestamp)
            for item in limited_episodes
        ]
        evidence_dicts = [item.to_dict() for item in limited_evidence]

        return {
            "selection": selection,
            "video_metadata": dict(self.knowledge.video_metadata),
            "time_ranges": {
                "requested": (
                    {
                        "start_timestamp": start_timestamp,
                        "end_timestamp": end_timestamp,
                    }
                    if time_filtered
                    else None
                ),
                "visual_observed": _observed_range(
                    [item.timestamp for item in limited_events]
                ),
                "text_observed": _observed_range(
                    [item.timestamp for item in limited_observations]
                    + [item.timestamp for item in limited_history_events]
                ),
            },
            "visual_objects": visual_objects,
            "visual_events": event_dicts,
            "episodes": visual_episodes,
            "text_observations": text_observation_dicts,
            "text_history": {
                "episodes": text_history_episode_dicts,
                "events": text_history_event_dicts,
            },
            "evidence_references": evidence_dicts,
            "truncated": dict(sorted(self._truncated.items())),
        }

    def _limit(self, items: list[Any], name: str, limit: int) -> list[Any]:
        omitted = max(0, len(items) - limit)
        if omitted:
            self._truncated[name] = self._truncated.get(name, 0) + omitted
        return items[:limit]

    def _object_dict(
        self,
        item: ObjectVideoFacts,
        *,
        time_filtered: bool,
        start_timestamp: float | None,
        end_timestamp: float | None,
    ) -> dict[str, object]:
        events = [
            event
            for event in item.events
            if not time_filtered
            or _in_range(event.timestamp, start_timestamp, end_timestamp)
        ]
        events = _ordered(events, lambda event: event.timestamp)
        events = self._limit(events, "object_events", self.limits.max_nested_events)
        if time_filtered:
            return {
                "track_id": item.track_id,
                "label": item.label,
                "events": [semantic_event_to_dict(event) for event in events],
            }
        result = item.to_dict()
        result["events"] = [semantic_event_to_dict(event) for event in events]
        return result

    def _episode_dict(
        self,
        episode: Any,
        start_timestamp: float | None,
        end_timestamp: float | None,
    ) -> dict[str, object]:
        events = [
            event
            for event in episode.events
            if _in_range(event.timestamp, start_timestamp, end_timestamp)
        ]
        events = _ordered(events, lambda event: event.timestamp)
        nested_events = self._limit(
            events,
            "episode_events",
            self.limits.max_nested_events,
        )
        relevant_evidence = {
            item
            for event in nested_events
            for item in self.knowledge.evidence_index.by_event(event)
        }
        evidence = [item for item in episode.evidence if item in relevant_evidence]
        evidence = _ordered(evidence, lambda item: item.timestamp)
        nested_evidence = self._limit(
            evidence,
            "episode_evidence",
            self.limits.max_nested_evidence,
        )
        positions: list[tuple[float, float]] = []
        confidences: list[float] = []
        for event in nested_events:
            for position in (event.previous_position, event.current_position):
                if position is not None and (not positions or positions[-1] != position):
                    positions.append(position)
            if event.confidence is not None:
                confidences.append(event.confidence)
        event_dicts = [semantic_event_to_dict(event) for event in nested_events]
        clipped = len(nested_events) < len(episode.events)
        return {
            "episode_type": episode.episode_type,
            "track_id": episode.track_id,
            "label": episode.label,
            "start_timestamp": (
                min(event.timestamp for event in nested_events)
                if nested_events
                else episode.start_timestamp
            ),
            "end_timestamp": (
                episode.end_timestamp
                if episode.end_timestamp is not None
                and _in_range(episode.end_timestamp, start_timestamp, end_timestamp)
                else None
            ),
            "event_types": [event.event_type.value for event in nested_events],
            "events": event_dicts,
            "positions": [list(item) for item in positions],
            "confidences": confidences,
            "evidence": [item.to_dict() for item in nested_evidence],
            "incomplete": episode.incomplete or clipped,
        }

    def _text_episode_dict(
        self,
        episode: VideoTextEpisode,
        *,
        time_filtered: bool,
        start_timestamp: float | None,
        end_timestamp: float | None,
        selected_events: list[VideoTextEvent],
    ) -> dict[str, object]:
        observations = [
            item
            for item in episode.observations
            if not time_filtered
            or _in_range(item.timestamp, start_timestamp, end_timestamp)
        ]
        observations = _ordered(observations, lambda item: item.timestamp)
        observations = self._limit(
            observations,
            "text_episode_observations",
            self.limits.max_nested_text_observations,
        )
        result = episode.to_dict()
        result["observations"] = [item.to_dict() for item in observations]
        if time_filtered:
            disappearance = next(
                (
                    item
                    for item in selected_events
                    if item.source_id == episode.source_id
                    and item.normalized_text == episode.normalized_text
                    and item.event_type.value == "TEXT_DISAPPEARED"
                ),
                None,
            )
            result["first_timestamp"] = (
                min(item.timestamp for item in observations) if observations else None
            )
            result["last_timestamp"] = (
                max(item.timestamp for item in observations) if observations else None
            )
            result["frame_indexes"] = [
                item.frame_index for item in observations if item.frame_index is not None
            ]
            result["confidences"] = [
                item.confidence for item in observations if item.confidence is not None
            ]
            result["bounding_boxes"] = [
                list(item.bbox) for item in observations if item.bbox is not None
            ]
            result["is_present"] = None
            result["disappeared_timestamp"] = (
                disappearance.timestamp if disappearance is not None else None
            )
            result["disappeared_frame_index"] = (
                disappearance.frame_index if disappearance is not None else None
            )
        return result


def _in_range(
    timestamp: float,
    start_timestamp: float | None,
    end_timestamp: float | None,
) -> bool:
    return (
        (start_timestamp is None or timestamp >= start_timestamp)
        and (end_timestamp is None or timestamp <= end_timestamp)
    )


def _ordered(items: list[Any], timestamp) -> list[Any]:
    return [item for _, item in sorted(enumerate(items), key=lambda pair: (timestamp(pair[1]), pair[0]))]


def _observed_range(timestamps: list[float]) -> dict[str, float] | None:
    if not timestamps:
        return None
    return {
        "start_timestamp": min(timestamps),
        "end_timestamp": max(timestamps),
    }


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _event_in_text_episode(
    episode: VideoTextEpisode,
    event: VideoTextEvent,
) -> bool:
    end_timestamp = (
        episode.disappeared_timestamp
        if episode.disappeared_timestamp is not None
        else episode.last_timestamp
    )
    return episode.first_timestamp <= event.timestamp <= end_timestamp


def _validate_range(start_timestamp: float, end_timestamp: float) -> None:
    if not isfinite(start_timestamp) or not isfinite(end_timestamp):
        raise ValueError("timestamps must be finite")
    if start_timestamp > end_timestamp:
        raise ValueError("start_timestamp must not be after end_timestamp")
