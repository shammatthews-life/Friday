from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
import math

from src.perception.video.ocr import VideoTextObservation


class VideoTextEventType(str, Enum):
    APPEARED = "TEXT_APPEARED"
    REMAINED = "TEXT_REMAINED"
    DISAPPEARED = "TEXT_DISAPPEARED"
    REAPPEARED = "TEXT_REAPPEARED"


@dataclass(frozen=True)
class VideoTextEvent:
    event_type: VideoTextEventType
    normalized_text: str
    text: str
    timestamp: float
    frame_index: int | None
    confidence: float | None
    bbox: tuple[float, float, float, float] | None
    source_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type.value,
            "normalized_text": self.normalized_text,
            "text": self.text,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "confidence": self.confidence,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class VideoTextEpisode:
    normalized_text: str
    text: str
    source_id: str | None
    first_timestamp: float
    last_timestamp: float
    is_present: bool
    disappeared_timestamp: float | None
    disappeared_frame_index: int | None
    observations: tuple[VideoTextObservation, ...]

    @property
    def frame_indexes(self) -> tuple[int, ...]:
        return tuple(
            observation.frame_index
            for observation in self.observations
            if observation.frame_index is not None
        )

    @property
    def confidences(self) -> tuple[float, ...]:
        return tuple(
            observation.confidence
            for observation in self.observations
            if observation.confidence is not None
        )

    @property
    def bounding_boxes(self) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(
            observation.bbox
            for observation in self.observations
            if observation.bbox is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "normalized_text": self.normalized_text,
            "text": self.text,
            "source_id": self.source_id,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "frame_indexes": list(self.frame_indexes),
            "confidences": list(self.confidences),
            "bounding_boxes": [list(box) for box in self.bounding_boxes],
            "is_present": self.is_present,
            "disappeared_timestamp": self.disappeared_timestamp,
            "disappeared_frame_index": self.disappeared_frame_index,
            "observations": [item.to_dict() for item in self.observations],
        }


@dataclass
class _EpisodeState:
    normalized_text: str
    text: str
    source_id: str | None
    first_timestamp: float
    last_timestamp: float
    observations: deque[VideoTextObservation]
    is_present: bool = True
    disappeared_timestamp: float | None = None
    disappeared_frame_index: int | None = None

    def snapshot(self) -> VideoTextEpisode:
        return VideoTextEpisode(
            normalized_text=self.normalized_text,
            text=self.text,
            source_id=self.source_id,
            first_timestamp=self.first_timestamp,
            last_timestamp=self.last_timestamp,
            is_present=self.is_present,
            disappeared_timestamp=self.disappeared_timestamp,
            disappeared_frame_index=self.disappeared_frame_index,
            observations=tuple(self.observations),
        )


class VideoTextHistory:
    """Group observed OCR text into bounded, per-source presence episodes."""

    def __init__(
        self,
        *,
        history_size: int = 256,
        max_frame_gap: int = 15,
        max_time_gap_seconds: float = 2.0,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        if max_frame_gap < 0:
            raise ValueError("max_frame_gap cannot be negative")
        if not math.isfinite(max_time_gap_seconds) or max_time_gap_seconds < 0:
            raise ValueError("max_time_gap_seconds must be finite and non-negative")
        self.history_size = history_size
        self.max_frame_gap = max_frame_gap
        self.max_time_gap_seconds = max_time_gap_seconds
        self._episodes: deque[_EpisodeState] = deque(maxlen=history_size)
        self._active: OrderedDict[tuple[str | None, str], _EpisodeState] = OrderedDict()
        self._last_by_key: OrderedDict[tuple[str | None, str], _EpisodeState] = OrderedDict()
        self._events: deque[VideoTextEvent] = deque(maxlen=history_size)
        self._last_timestamp_by_source: OrderedDict[str | None, float] = OrderedDict()

    @property
    def episodes(self) -> tuple[VideoTextEpisode, ...]:
        return tuple(
            episode.snapshot()
            for episode in sorted(
                self._episodes,
                key=lambda item: (item.first_timestamp, item.source_id or "", item.normalized_text),
            )
        )

    @property
    def events(self) -> tuple[VideoTextEvent, ...]:
        return tuple(
            sorted(
                self._events,
                key=lambda item: (
                    item.timestamp,
                    -1 if item.frame_index is None else item.frame_index,
                    item.source_id or "",
                    item.normalized_text,
                ),
            )
        )

    def add_frame(
        self,
        *,
        timestamp: float,
        frame_index: int | None,
        source_id: str | None,
        observations: tuple[VideoTextObservation, ...] | list[VideoTextObservation],
    ) -> tuple[VideoTextEvent, ...]:
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        previous_timestamp = self._last_timestamp_by_source.get(source_id)
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError("frames must be processed in chronological order per source")
        self._last_timestamp_by_source[source_id] = timestamp
        self._last_timestamp_by_source.move_to_end(source_id)
        while len(self._last_timestamp_by_source) > self.history_size:
            self._last_timestamp_by_source.popitem(last=False)

        emitted: list[VideoTextEvent] = []
        for key, episode in tuple(self._active.items()):
            if key[0] == source_id and not self._within_gap(
                episode,
                timestamp=timestamp,
                frame_index=frame_index,
            ):
                episode.is_present = False
                episode.disappeared_timestamp = timestamp
                episode.disappeared_frame_index = frame_index
                self._active.pop(key)
                event = VideoTextEvent(
                    event_type=VideoTextEventType.DISAPPEARED,
                    normalized_text=episode.normalized_text,
                    text=episode.text,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    confidence=None,
                    bbox=None,
                    source_id=episode.source_id,
                )
                self._events.append(event)
                emitted.append(event)

        seen_this_frame: set[tuple[str | None, str]] = set()
        for observation in observations:
            normalized = _normalize(observation.text)
            if not normalized:
                continue
            observation_source = (
                observation.source_id
                if observation.source_id is not None
                else source_id
            )
            key = (observation_source, normalized)
            if key in seen_this_frame:
                continue
            seen_this_frame.add(key)
            episode = self._active.get(key)
            if episode is None:
                previous = self._last_by_key.get(key)
                event_type = (
                    VideoTextEventType.REAPPEARED
                    if previous is not None and not previous.is_present
                    else VideoTextEventType.APPEARED
                )
                episode = _EpisodeState(
                    normalized_text=normalized,
                    text=observation.text,
                    source_id=observation_source,
                    first_timestamp=observation.timestamp,
                    last_timestamp=observation.timestamp,
                    observations=deque(maxlen=self.history_size),
                )
                evicted = self._episodes[0] if len(self._episodes) == self.history_size else None
                self._episodes.append(episode)
                if evicted is not None:
                    evicted_key = (evicted.source_id, evicted.normalized_text)
                    if self._last_by_key.get(evicted_key) is evicted:
                        self._last_by_key.pop(evicted_key)
                    if self._active.get(evicted_key) is evicted:
                        self._active.pop(evicted_key)
                self._active[key] = episode
                self._last_by_key[key] = episode
                while len(self._active) > self.history_size:
                    self._active.popitem(last=False)
                while len(self._last_by_key) > self.history_size:
                    self._last_by_key.popitem(last=False)
            else:
                event_type = VideoTextEventType.REMAINED
                episode.last_timestamp = observation.timestamp
                episode.disappeared_timestamp = None
                episode.disappeared_frame_index = None
                self._active.move_to_end(key)
                self._last_by_key.move_to_end(key)

            episode.observations.append(observation)
            event = VideoTextEvent(
                event_type=event_type,
                normalized_text=normalized,
                text=observation.text,
                timestamp=observation.timestamp,
                frame_index=observation.frame_index,
                confidence=observation.confidence,
                bbox=observation.bbox,
                source_id=observation_source,
            )
            self._events.append(event)
            emitted.append(event)
        return tuple(emitted)

    def texts_present(self, *, source_id: str | None = None) -> tuple[VideoTextEpisode, ...]:
        return tuple(
            episode.snapshot()
            for (item_source, _), episode in self._active.items()
            if source_id is None or item_source == source_id
        )

    def texts_appeared(
        self,
        *,
        source_id: str | None = None,
    ) -> tuple[VideoTextEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.event_type in (VideoTextEventType.APPEARED, VideoTextEventType.REAPPEARED)
            and (source_id is None or event.source_id == source_id)
        )

    def texts_disappeared(
        self,
        *,
        source_id: str | None = None,
    ) -> tuple[VideoTextEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.event_type is VideoTextEventType.DISAPPEARED
            and (source_id is None or event.source_id == source_id)
        )

    def text_history(
        self,
        text: str,
        *,
        source_id: str | None = None,
    ) -> tuple[VideoTextEvent, ...]:
        normalized = _normalize(text)
        return tuple(
            event
            for event in self.events
            if event.normalized_text == normalized
            and (source_id is None or event.source_id == source_id)
        )

    def texts_in_time_range(
        self,
        start_timestamp: float,
        end_timestamp: float,
        *,
        source_id: str | None = None,
    ) -> tuple[VideoTextEvent, ...]:
        if start_timestamp > end_timestamp:
            raise ValueError("start_timestamp must not be after end_timestamp")
        return tuple(
            event
            for event in self.events
            if start_timestamp <= event.timestamp <= end_timestamp
            and (source_id is None or event.source_id == source_id)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "episodes": [episode.to_dict() for episode in self.episodes],
            "events": [event.to_dict() for event in self.events],
        }

    def _within_gap(
        self,
        episode: _EpisodeState,
        *,
        timestamp: float,
        frame_index: int | None,
    ) -> bool:
        elapsed = timestamp - episode.last_timestamp
        if elapsed < 0 or elapsed > self.max_time_gap_seconds:
            return False
        previous_index = (
            episode.observations[-1].frame_index if episode.observations else None
        )
        if previous_index is not None and frame_index is not None:
            frame_gap = frame_index - previous_index
            return 0 <= frame_gap <= self.max_frame_gap
        return True


class VideoTextQueryEngine:
    """Structured query facade for text-history results."""

    def __init__(self, history: VideoTextHistory) -> None:
        self.history = history

    def texts_present(self, *, source_id: str | None = None) -> tuple[VideoTextEpisode, ...]:
        return self.history.texts_present(source_id=source_id)

    def texts_appeared(self, *, source_id: str | None = None) -> tuple[VideoTextEvent, ...]:
        return self.history.texts_appeared(source_id=source_id)

    def texts_disappeared(
        self,
        *,
        source_id: str | None = None,
    ) -> tuple[VideoTextEvent, ...]:
        return self.history.texts_disappeared(source_id=source_id)

    def text_history(
        self,
        text: str,
        *,
        source_id: str | None = None,
    ) -> tuple[VideoTextEvent, ...]:
        return self.history.text_history(text, source_id=source_id)

    def texts_in_time_range(
        self,
        start_timestamp: float,
        end_timestamp: float,
        *,
        source_id: str | None = None,
    ) -> tuple[VideoTextEvent, ...]:
        return self.history.texts_in_time_range(
            start_timestamp,
            end_timestamp,
            source_id=source_id,
        )


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())
