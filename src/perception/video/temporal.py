from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.perception.types import PerceptionFrame


@dataclass(frozen=True)
class FrameChange:
    previous_frame_index: int | None
    frame_index: int | None
    previous_timestamp: float
    timestamp: float
    mean_difference: float
    changed_fraction: float
    meaningful: bool


@dataclass(frozen=True)
class TemporalFrameSample:
    frame: PerceptionFrame
    sample_index: int
    elapsed_seconds: float
    elapsed_frames: int
    is_keyframe: bool
    selection_reason: str
    change: FrameChange | None


class FrameChangeDetector:
    """Compare frames using deterministic, downsampled pixel differences."""

    def __init__(
        self,
        *,
        mean_difference_threshold: float = 0.08,
        changed_fraction_threshold: float = 0.10,
        per_pixel_threshold: float = 0.15,
        thumbnail_size: int = 32,
    ) -> None:
        if not 0 <= mean_difference_threshold <= 1:
            raise ValueError("mean_difference_threshold must be between 0 and 1")
        if not 0 <= changed_fraction_threshold <= 1:
            raise ValueError("changed_fraction_threshold must be between 0 and 1")
        if not 0 <= per_pixel_threshold <= 1:
            raise ValueError("per_pixel_threshold must be between 0 and 1")
        if thumbnail_size < 1:
            raise ValueError("thumbnail_size must be at least 1")
        self.mean_difference_threshold = mean_difference_threshold
        self.changed_fraction_threshold = changed_fraction_threshold
        self.per_pixel_threshold = per_pixel_threshold
        self.thumbnail_size = thumbnail_size

    def _thumbnail(self, image: np.ndarray) -> np.ndarray:
        if (
            not isinstance(image, np.ndarray)
            or image.ndim not in (2, 3)
            or image.size == 0
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError("Frame change detection requires a non-empty 2D or 3D image")
        if not np.issubdtype(image.dtype, np.number):
            raise ValueError("Frame change detection requires numeric image data")

        rows = np.linspace(0, image.shape[0] - 1, min(image.shape[0], self.thumbnail_size), dtype=int)
        columns = np.linspace(0, image.shape[1] - 1, min(image.shape[1], self.thumbnail_size), dtype=int)
        thumbnail = image[rows[:, None], columns]
        values = thumbnail.astype(np.float32)
        if np.issubdtype(image.dtype, np.integer):
            values /= float(np.iinfo(image.dtype).max)
        elif values.size and float(np.nanmax(values)) > 1.0:
            values /= 255.0
        return np.clip(values, 0.0, 1.0)

    def compare(self, previous: PerceptionFrame, current: PerceptionFrame) -> FrameChange:
        previous_pixels = self._thumbnail(previous.image)
        current_pixels = self._thumbnail(current.image)
        if previous_pixels.shape != current_pixels.shape:
            raise ValueError("Frame images must have matching dimensions and channel counts")

        difference = np.abs(current_pixels - previous_pixels)
        mean_difference = float(np.mean(difference))
        changed_fraction = float(np.mean(difference >= self.per_pixel_threshold))
        meaningful = (
            mean_difference >= self.mean_difference_threshold
            and changed_fraction >= self.changed_fraction_threshold
        )
        return FrameChange(
            previous_frame_index=previous.frame_index,
            frame_index=current.frame_index,
            previous_timestamp=previous.timestamp,
            timestamp=current.timestamp,
            mean_difference=mean_difference,
            changed_fraction=changed_fraction,
            meaningful=meaningful,
        )


class KeyframeSelector:
    """Select the first frame, meaningful changes, and periodic representatives."""

    def __init__(
        self,
        *,
        max_interval_frames: int = 30,
        change_detector: FrameChangeDetector | None = None,
    ) -> None:
        if max_interval_frames < 1:
            raise ValueError("max_interval_frames must be at least 1")
        self.max_interval_frames = max_interval_frames
        self.change_detector = change_detector or FrameChangeDetector()
        self._previous: PerceptionFrame | None = None
        self._last_keyframe_sample: int | None = None

    def consider(self, frame: PerceptionFrame, sample_index: int) -> tuple[bool, str, FrameChange | None]:
        change = (
            self.change_detector.compare(self._previous, frame)
            if self._previous is not None
            else None
        )
        if self._previous is None:
            selected, reason = True, "first_frame"
        elif change is not None and change.meaningful:
            selected, reason = True, "meaningful_change"
        elif (
            self._last_keyframe_sample is not None
            and sample_index - self._last_keyframe_sample >= self.max_interval_frames
        ):
            selected, reason = True, "maximum_interval"
        else:
            selected, reason = False, "no_significant_change"

        if selected:
            self._last_keyframe_sample = sample_index
        self._previous = frame
        return selected, reason, change


class TemporalFrameBuffer:
    """A fixed-capacity, oldest-first buffer of recent temporal samples."""

    def __init__(self, capacity: int = 12) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._samples: deque[TemporalFrameSample] = deque(maxlen=capacity)

    def append(self, sample: TemporalFrameSample) -> None:
        self._samples.append(sample)

    def snapshot(self) -> tuple[TemporalFrameSample, ...]:
        return tuple(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


class TemporalSampler:
    """Attach timing and keyframe metadata while retaining a bounded frame history."""

    def __init__(
        self,
        *,
        selector: KeyframeSelector | None = None,
        buffer_capacity: int = 12,
    ) -> None:
        self.selector = selector or KeyframeSelector()
        self.buffer = TemporalFrameBuffer(buffer_capacity)
        self._sample_count = 0
        self._previous: PerceptionFrame | None = None

    def process(self, frame: PerceptionFrame) -> TemporalFrameSample:
        sample_index = self._sample_count
        self._sample_count += 1
        if self._previous is None:
            elapsed_seconds = 0.0
            elapsed_frames = 0
        else:
            elapsed_seconds = max(0.0, frame.timestamp - self._previous.timestamp)
            if frame.frame_index is not None and self._previous.frame_index is not None:
                elapsed_frames = max(0, frame.frame_index - self._previous.frame_index)
            else:
                elapsed_frames = 1

        is_keyframe, selection_reason, change = self.selector.consider(frame, sample_index)
        sample = TemporalFrameSample(
            frame=frame,
            sample_index=sample_index,
            elapsed_seconds=elapsed_seconds,
            elapsed_frames=elapsed_frames,
            is_keyframe=is_keyframe,
            selection_reason=selection_reason,
            change=change,
        )
        self.buffer.append(sample)
        self._previous = frame
        return sample
