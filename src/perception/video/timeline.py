from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.perception.types import SceneSnapshot, TrackEvent, TrackedObject


class SemanticEventType(str, Enum):
    OBJECT_APPEARED = "OBJECT_APPEARED"
    OBJECT_REMAINED = "OBJECT_REMAINED"
    OBJECT_MOVED = "OBJECT_MOVED"
    OBJECT_DISAPPEARED = "OBJECT_DISAPPEARED"
    OBJECT_REACQUIRED = "OBJECT_REACQUIRED"


@dataclass(frozen=True)
class SemanticEvent:
    event_type: SemanticEventType
    timestamp: float
    frame_index: int | None
    track_id: int | None
    label: str
    previous_position: tuple[float, float] | None = None
    current_position: tuple[float, float] | None = None
    confidence: float | None = None


class SemanticTimeline:
    """Bounded chronological events derived from semantic tracker snapshots."""

    _TRACK_EVENT_TYPES = {
        "appeared": SemanticEventType.OBJECT_APPEARED,
        "moved": SemanticEventType.OBJECT_MOVED,
        "disappeared": SemanticEventType.OBJECT_DISAPPEARED,
        "reacquired": SemanticEventType.OBJECT_REACQUIRED,
    }

    def __init__(self, history_size: int = 256) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        self.history_size = history_size
        self._events: list[SemanticEvent] = []
        self._last_positions: dict[int, tuple[float, float]] = {}
        self._remained_emitted: set[int] = set()
        self._last_observation_key: tuple[str, int | None, float] | None = None

    @property
    def events(self) -> tuple[SemanticEvent, ...]:
        return tuple(self._events)

    def add_observation(
        self,
        snapshot: SceneSnapshot,
        *,
        frame_index: int | None = None,
    ) -> tuple[SemanticEvent, ...]:
        """Consume one perception snapshot and return events emitted for it."""
        if not snapshot.valid:
            raise ValueError("Cannot add an invalid scene snapshot to the timeline")

        observation_key = (snapshot.source_id, frame_index, snapshot.timestamp)
        if observation_key == self._last_observation_key:
            return ()
        self._last_observation_key = observation_key

        tracks_by_id = {track.track_id: track for track in snapshot.tracks}
        track_events = snapshot.scene_changes
        if not track_events:
            track_events = snapshot.newly_detected + snapshot.disappeared

        emitted: list[SemanticEvent] = []
        transitioned_tracks: set[int] = set()
        for track_event in track_events:
            event_type = self._TRACK_EVENT_TYPES.get(track_event.kind.lower())
            if event_type is None:
                continue
            transitioned_tracks.add(track_event.track_id)
            track = tracks_by_id.get(track_event.track_id)
            current_position = self._position(track)
            previous_position = self._last_positions.get(track_event.track_id)
            if previous_position is None and track_event.previous_box is not None:
                previous_position = self._box_center(track_event.previous_box)
            if current_position is None and track_event.current_box is not None:
                current_position = self._box_center(track_event.current_box)

            emitted.append(
                SemanticEvent(
                    event_type=event_type,
                    timestamp=track_event.timestamp,
                    frame_index=frame_index,
                    track_id=track_event.track_id,
                    label=track_event.label,
                    previous_position=previous_position,
                    current_position=current_position,
                    confidence=track.confidence if track is not None else None,
                )
            )
            if event_type in {
                SemanticEventType.OBJECT_APPEARED,
                SemanticEventType.OBJECT_MOVED,
                SemanticEventType.OBJECT_REACQUIRED,
            }:
                self._remained_emitted.discard(track_event.track_id)
            elif event_type is SemanticEventType.OBJECT_DISAPPEARED:
                self._remained_emitted.discard(track_event.track_id)

        for track in snapshot.tracks:
            if track.state == "remained" and track.track_id not in transitioned_tracks:
                if track.track_id not in self._remained_emitted:
                    emitted.append(
                        SemanticEvent(
                            event_type=SemanticEventType.OBJECT_REMAINED,
                            timestamp=snapshot.timestamp,
                            frame_index=frame_index,
                            track_id=track.track_id,
                            label=track.label,
                            previous_position=self._last_positions.get(track.track_id),
                            current_position=self._position(track),
                            confidence=track.confidence,
                        )
                    )
                    self._remained_emitted.add(track.track_id)

        for track in snapshot.tracks:
            self._last_positions[track.track_id] = self._position(track)
        active_ids = set(tracks_by_id)
        self._last_positions = {
            track_id: position
            for track_id, position in self._last_positions.items()
            if track_id in active_ids
        }
        self._remained_emitted.intersection_update(active_ids)
        self._append(emitted)
        return tuple(emitted)

    def _append(self, events: list[SemanticEvent]) -> None:
        if not events:
            return
        self._events.extend(events)
        self._events.sort(key=lambda event: event.timestamp)
        self._events = self._events[-self.history_size :]

    @staticmethod
    def _position(track: TrackedObject | None) -> tuple[float, float] | None:
        if track is None:
            return None
        return (track.normalized_horizontal, track.normalized_vertical)

    @staticmethod
    def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
