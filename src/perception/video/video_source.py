from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from src.perception.types import PerceptionFrame


class VideoInputError(RuntimeError):
    """Raised when a local video file cannot be opened for decoding."""


CaptureFactory = Callable[[str], Any]


class VideoFrameSource(Iterator[PerceptionFrame]):
    """Decode a local video file into sampled PerceptionFrame objects.

    ``target_fps`` takes precedence when supplied. Otherwise every
    ``every_n_frames`` decoded source frames are yielded, starting at index 0.
    Decoding and sampling are independent of perception inference.
    """

    _CAP_PROP_POS_MSEC = 0
    _CAP_PROP_FPS = 5

    def __init__(
        self,
        video_path: str | Path,
        *,
        every_n_frames: int = 1,
        target_fps: float | None = None,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        if every_n_frames < 1:
            raise ValueError("every_n_frames must be at least 1")
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        self.every_n_frames = int(every_n_frames)
        self.target_fps = float(target_fps) if target_fps is not None else None
        self.capture_factory = capture_factory
        self._capture: Any | None = None
        self._source_index = 0
        self._last_timestamp: float | None = None
        self._next_target_timestamp = 0.0
        self._ended = False
        self._closed = False
        self.fps: float | None = None
        self.decoded_frames = 0
        self.yielded_frames = 0
        self.skipped_invalid_frames = 0
        self.last_error: str | None = None

    def open(self) -> None:
        if self._capture is not None:
            return
        if self._closed:
            raise VideoInputError("Video source is already closed")
        if not self.video_path.is_file():
            raise VideoInputError(f"Video file does not exist: {self.video_path}")
        factory = self.capture_factory
        if factory is None:
            import cv2

            factory = lambda path: cv2.VideoCapture(path)
        capture = factory(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise VideoInputError(f"Could not open local video file: {self.video_path}")
        self._capture = capture
        fps = self._get_property(self._CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and fps > 0 else None

    def _get_property(self, property_id: int) -> float:
        try:
            return float(self._capture.get(property_id))
        except (AttributeError, TypeError, ValueError, OSError):
            return float("nan")

    def _timestamp_for(self, source_index: int) -> float:
        position_ms = self._get_property(self._CAP_PROP_POS_MSEC)
        if math.isfinite(position_ms) and position_ms >= 0:
            timestamp = position_ms / 1000.0
        elif self.fps is not None:
            timestamp = source_index / self.fps
        else:
            timestamp = source_index / 30.0
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            step = 1.0 / self.fps if self.fps else 1.0 / 30.0
            timestamp = self._last_timestamp + step
        self._last_timestamp = timestamp
        return timestamp

    def _selected(self, source_index: int, timestamp: float) -> bool:
        if self.target_fps is None:
            return source_index % self.every_n_frames == 0
        if timestamp + 1e-9 < self._next_target_timestamp:
            return False
        self._next_target_timestamp = timestamp + 1.0 / self.target_fps
        return True

    def __iter__(self) -> VideoFrameSource:
        self.open()
        return self

    def __next__(self) -> PerceptionFrame:
        if self._ended or self._closed:
            raise StopIteration
        self.open()
        while True:
            try:
                ok, image = self._capture.read()
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                self._ended = True
                self.close()
                raise StopIteration from error
            if not ok:
                self._ended = True
                self.close()
                raise StopIteration

            source_index = self._source_index
            self._source_index += 1
            self.decoded_frames += 1
            timestamp = self._timestamp_for(source_index)
            if (
                not isinstance(image, np.ndarray)
                or image.ndim != 3
                or image.shape[0] <= 0
                or image.shape[1] <= 0
                or image.size == 0
            ):
                self.skipped_invalid_frames += 1
                continue
            if not self._selected(source_index, timestamp):
                continue

            frame = PerceptionFrame(
                image=image,
                timestamp=timestamp,
                source_id=f"video:{self.video_path.name}",
                frame_index=source_index,
            )
            self.yielded_frames += 1
            return frame

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
        self._closed = True

    def __enter__(self) -> VideoFrameSource:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
