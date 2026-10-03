from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import math
from typing import Iterable, Protocol

import numpy as np

from src.perception.types import PerceptionFrame


@dataclass(frozen=True)
class OCRDetection:
    """Text and optional metadata returned directly by an OCR backend."""

    text: str
    confidence: float | None = None
    bbox: tuple[float, float, float, float] | None = None


class OCRBackend(Protocol):
    """Backend contract: inspect an image and return only observed OCR metadata."""

    def extract(self, image: np.ndarray) -> Iterable[OCRDetection]:
        ...


class OCRBackendError(RuntimeError):
    """Raised by an OCR backend when it cannot extract text from an image."""


@dataclass(frozen=True)
class VideoTextObservation:
    text: str
    confidence: float | None
    timestamp: float
    frame_index: int | None
    bbox: tuple[float, float, float, float] | None
    source_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "source_id": self.source_id,
        }


class VideoTextExtractor:
    """Apply OCR to frames, suppress nearby repeated text, and bound retained history."""

    def __init__(
        self,
        backend: OCRBackend,
        *,
        history_size: int = 256,
        dedup_frame_gap: int = 15,
        dedup_time_gap_seconds: float = 2.0,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        if dedup_frame_gap < 0:
            raise ValueError("dedup_frame_gap cannot be negative")
        if dedup_time_gap_seconds < 0:
            raise ValueError("dedup_time_gap_seconds cannot be negative")
        self.backend = backend
        self.history_size = history_size
        self.dedup_frame_gap = dedup_frame_gap
        self.dedup_time_gap_seconds = dedup_time_gap_seconds
        self._observations: deque[VideoTextObservation] = deque(maxlen=history_size)
        self._last_seen: OrderedDict[tuple[str | None, str], VideoTextObservation] = OrderedDict()
        self.duplicates_suppressed = 0

    @property
    def observations(self) -> tuple[VideoTextObservation, ...]:
        return tuple(self._observations)

    def process(self, frame: PerceptionFrame) -> tuple[VideoTextObservation, ...]:
        emitted: list[VideoTextObservation] = []
        detections = self.backend.extract(frame.image)
        for detection in detections:
            if not isinstance(detection, OCRDetection):
                raise TypeError("OCR backend must return OCRDetection values")
            if not _normalize_text(detection.text):
                continue
            observation = VideoTextObservation(
                text=detection.text,
                confidence=detection.confidence,
                timestamp=frame.timestamp,
                frame_index=frame.frame_index,
                bbox=detection.bbox,
                source_id=frame.source_id or None,
            )
            key = (observation.source_id, _normalize_text(observation.text))
            previous = self._last_seen.get(key)
            duplicate = previous is not None and _is_nearby(
                previous,
                observation,
                max_frame_gap=self.dedup_frame_gap,
                max_time_gap=self.dedup_time_gap_seconds,
            )
            self._last_seen[key] = observation
            self._last_seen.move_to_end(key)
            while len(self._last_seen) > self.history_size:
                self._last_seen.popitem(last=False)

            if duplicate:
                self.duplicates_suppressed += 1
                continue
            self._observations.append(observation)
            emitted.append(observation)
        return tuple(emitted)


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _is_nearby(
    previous: VideoTextObservation,
    current: VideoTextObservation,
    *,
    max_frame_gap: int,
    max_time_gap: float,
) -> bool:
    if current.timestamp < previous.timestamp:
        return False
    elapsed = current.timestamp - previous.timestamp
    if not math.isfinite(elapsed) or elapsed > max_time_gap:
        return False
    if previous.frame_index is not None and current.frame_index is not None:
        frame_gap = current.frame_index - previous.frame_index
        return 0 <= frame_gap <= max_frame_gap
    return True
