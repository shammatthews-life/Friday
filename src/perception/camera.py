from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterator

from src.perception.frame import validate_frame
from src.perception.types import PerceptionFrame


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    backend: str = "CAP_DSHOW"
    warmup_seconds: float = 1.0
    minimum_brightness: float = 8.0
    minimum_contrast: float = 1.0


class CameraSource:
    """Bounded DirectShow frame source with warm-up, validation, and cleanup."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._capture = None
        self._frame_index = 0
        self.last_rejection: str | None = None

    def open(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            return
        import cv2

        backends = {"CAP_DSHOW": cv2.CAP_DSHOW, "CAP_ANY": cv2.CAP_ANY}
        if self.config.backend not in backends:
            raise ValueError(f"Unsupported camera backend: {self.config.backend}")
        capture = cv2.VideoCapture(self.config.index, backends[self.config.backend])
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open camera index {self.config.index} with {self.config.backend}"
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture = capture

    def warm_up(self) -> PerceptionFrame:
        self.open()
        deadline = time.perf_counter() + max(0.0, self.config.warmup_seconds)
        latest_valid: PerceptionFrame | None = None
        first_read = True
        while first_read or time.perf_counter() < deadline:
            first_read = False
            frame = self.read()
            if frame is not None:
                latest_valid = frame
        if latest_valid is None:
            raise RuntimeError("Camera warm-up produced no usable frames")
        return latest_valid

    def read(self) -> PerceptionFrame | None:
        self.open()
        ok, image = self._capture.read()
        if not ok or image is None:
            self.last_rejection = "camera returned no frame"
            return None
        self._frame_index += 1
        frame = PerceptionFrame(
            image=image,
            timestamp=time.time(),
            source_id=f"camera:{self.config.index}",
            frame_index=self._frame_index,
        )
        validation = validate_frame(
            frame,
            minimum_brightness=self.config.minimum_brightness,
            minimum_contrast=self.config.minimum_contrast,
        )
        self.last_rejection = None if validation.valid else validation.reason
        return frame if validation.valid else None

    def frames(self, duration_seconds: float) -> Iterator[PerceptionFrame]:
        self.open()
        deadline = time.perf_counter() + max(0.0, duration_seconds)
        while time.perf_counter() < deadline:
            frame = self.read()
            if frame is not None:
                yield frame

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()

    def __enter__(self) -> CameraSource:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
