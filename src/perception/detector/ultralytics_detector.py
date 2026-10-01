from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import numpy as np

from src.perception.spatial.relations import horizontal_region, vertical_region
from src.perception.types import Detection, PerceptionFrame


class Detector(Protocol):
    def detect(self, frame: PerceptionFrame) -> list[Detection]:
        """Detect objects in a single image/video/camera frame."""


class PromptedDetector(Detector, Protocol):
    def set_text_prompts(self, prompts: list[str]) -> None:
        """Configure optional open-vocabulary text prompts."""

    def set_visual_prompts(self, frame: PerceptionFrame, boxes: list[tuple[float, float, float, float]]) -> None:
        """Configure optional visual prompts from existing detections."""


class UltralyticsDetector:
    """Lazy closed-set YOLO adapter for an already-local Ultralytics checkpoint."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        image_size: int = 640,
        confidence: float = 0.25,
        device: str = "cpu",
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.image_size = max(32, int(image_size))
        self.confidence = min(1.0, max(0.0, float(confidence)))
        self.device = device
        self.model_factory = model_factory
        self.model: Any | None = None
        self.load_time_ms: float | None = None
        self.last_inference_latency_ms: float | None = None
        self.inference_count = 0

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Local detector checkpoint not found: {self.model_path}")
        factory = self.model_factory
        if factory is None:
            from ultralytics import YOLO

            factory = YOLO
        started = time.perf_counter()
        self.model = factory(str(self.model_path))
        self.load_time_ms = (time.perf_counter() - started) * 1000

    def detect(self, frame: PerceptionFrame) -> list[Detection]:
        image = frame.image
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
            raise ValueError("Detector requires a non-empty three-dimensional image array")
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Detector received a zero-sized frame")
        self.load()
        started = time.perf_counter()
        results = self.model.predict(
            source=image,
            device=self.device,
            verbose=False,
            imgsz=self.image_size,
            conf=self.confidence,
        )
        self.last_inference_latency_ms = (time.perf_counter() - started) * 1000
        self.inference_count += 1
        if not results:
            return []
        result = results[0]
        detections: list[Detection] = []
        for box, confidence, class_id in zip(
            result.boxes.xyxy, result.boxes.conf, result.boxes.cls
        ):
            left, top, right, bottom = (float(value) for value in box)
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            normalized_x = center_x / width
            normalized_y = center_y / height
            detections.append(
                Detection(
                    label=str(result.names[int(class_id)]),
                    confidence=float(confidence),
                    bounding_box=(left, top, right, bottom),
                    center_x=center_x,
                    center_y=center_y,
                    normalized_horizontal=normalized_x,
                    normalized_vertical=normalized_y,
                    position_category=horizontal_region(normalized_x),
                    vertical_position=vertical_region(normalized_y),
                )
            )
        return detections

    def close(self) -> None:
        model = self.model
        self.model = None
        del model
        gc.collect()

    def __enter__(self) -> UltralyticsDetector:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
