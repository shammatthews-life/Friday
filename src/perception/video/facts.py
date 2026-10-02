from __future__ import annotations

from dataclasses import dataclass

from src.perception.video.timeline import SemanticEvent, SemanticEventType, SemanticTimeline


@dataclass(frozen=True)
class ObjectVideoFacts:
    track_id: int | None
    label: str
    first_seen_timestamp: float | None
    last_seen_timestamp: float | None
    presence_duration_seconds: float
    appearance_count: int
    reacquisition_count: int
    is_present: bool
    events: tuple[SemanticEvent, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "first_seen_timestamp": self.first_seen_timestamp,
            "last_seen_timestamp": self.last_seen_timestamp,
            "presence_duration_seconds": self.presence_duration_seconds,
            "appearance_count": self.appearance_count,
            "reacquisition_count": self.reacquisition_count,
            "is_present": self.is_present,
            "events": [_event_to_dict(event) for event in self.events],
        }


@dataclass(frozen=True)
class VideoFacts:
    """Machine-readable facts derived only from retained semantic timeline events."""

    events: tuple[SemanticEvent, ...]
    objects: tuple[ObjectVideoFacts, ...]

    @classmethod
    def from_timeline(
        cls,
        timeline: SemanticTimeline,
        *,
        history_size: int | None = None,
    ) -> VideoFacts:
        if history_size is not None and history_size < 1:
            raise ValueError("history_size must be at least 1")
        source_events = timeline.events
        retained_events = source_events[-history_size:] if history_size is not None else source_events
        ordered_events = tuple(
            event
            for _, event in sorted(
                enumerate(retained_events),
                key=lambda item: (item[1].timestamp, item[0]),
            )
        )
        grouped: dict[tuple[str, int], list[SemanticEvent]] = {}
        for index, event in enumerate(ordered_events):
            identity = ("track", event.track_id) if event.track_id is not None else ("event", index)
            grouped.setdefault(identity, []).append(event)
        objects = tuple(
            _derive_object_facts(track_events[0].track_id, track_events)
            for track_events in grouped.values()
        )
        return cls(events=ordered_events, objects=objects)

    @property
    def event_timeline(self) -> tuple[SemanticEvent, ...]:
        return self.events

    @property
    def objects_present(self) -> tuple[ObjectVideoFacts, ...]:
        return tuple(item for item in self.objects if item.is_present)

    @property
    def objects_appeared(self) -> tuple[ObjectVideoFacts, ...]:
        return tuple(item for item in self.objects if item.appearance_count > 0)

    @property
    def objects_disappeared(self) -> tuple[ObjectVideoFacts, ...]:
        return tuple(
            item
            for item in self.objects
            if any(event.event_type is SemanticEventType.OBJECT_DISAPPEARED for event in item.events)
        )

    @property
    def movement_events(self) -> tuple[SemanticEvent, ...]:
        return tuple(
            event for event in self.events
            if event.event_type is SemanticEventType.OBJECT_MOVED
        )

    def object_history(self, track_id: int) -> tuple[SemanticEvent, ...]:
        return tuple(event for event in self.events if event.track_id == track_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "events": [_event_to_dict(event) for event in self.events],
            "objects": [item.to_dict() for item in self.objects],
        }


def _derive_object_facts(
    track_id: int | None,
    events: list[SemanticEvent],
) -> ObjectVideoFacts:
    first_seen: float | None = None
    last_seen: float | None = None
    interval_start: float | None = None
    duration = 0.0
    appearance_count = 0
    reacquisition_count = 0
    is_present = False

    for event in events:
        if event.event_type in {
            SemanticEventType.OBJECT_APPEARED,
            SemanticEventType.OBJECT_REACQUIRED,
            SemanticEventType.OBJECT_REMAINED,
            SemanticEventType.OBJECT_MOVED,
        }:
            if first_seen is None:
                first_seen = event.timestamp
            last_seen = event.timestamp
            if not is_present:
                interval_start = event.timestamp
                is_present = True
            if event.event_type is SemanticEventType.OBJECT_APPEARED:
                appearance_count += 1
            elif event.event_type is SemanticEventType.OBJECT_REACQUIRED:
                reacquisition_count += 1
        elif event.event_type is SemanticEventType.OBJECT_DISAPPEARED:
            if is_present and interval_start is not None:
                duration += max(0.0, event.timestamp - interval_start)
            interval_start = None
            is_present = False

    if is_present and interval_start is not None and last_seen is not None:
        duration += max(0.0, last_seen - interval_start)

    return ObjectVideoFacts(
        track_id=track_id,
        label=events[0].label,
        first_seen_timestamp=first_seen,
        last_seen_timestamp=last_seen,
        presence_duration_seconds=duration,
        appearance_count=appearance_count,
        reacquisition_count=reacquisition_count,
        is_present=is_present,
        events=tuple(events),
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
