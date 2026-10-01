from __future__ import annotations

import gc
from pathlib import Path
import tempfile
from typing import Any, Callable, Protocol

import numpy as np

from src.perception.types import Detection, PerceptionFrame


class RelativeDepthProvider(Protocol):
    def estimate(self, frame: PerceptionFrame, detections: list[Detection]) -> dict[int, str]:
        """Return relative depth categories keyed by stable track id."""


class LazyDepthAnythingV2(RelativeDepthProvider):
    """On-demand adapter over the existing local Depth Anything V2 estimator."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        retain_model: bool = False,
        estimator_factory: Callable[[str | Path], Any] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.retain_model = retain_model
        self.estimator_factory = estimator_factory
        self.estimator: Any | None = None
        self.load_time_ms: float | None = None
        self.inference_latency_ms: float | None = None
        self.last_error: str | None = None

    def _get_estimator(self):
        if self.estimator is None:
            factory = self.estimator_factory
            if factory is None:
                from src.perception.depth.depth_estimator import DepthEstimator

                factory = DepthEstimator
            self.estimator = factory(self.model_dir)
        return self.estimator

    def estimate(self, frame: PerceptionFrame, detections: list[Detection]) -> dict[int, str]:
        if not isinstance(frame.image, np.ndarray) or frame.image.ndim != 3 or frame.image.size == 0:
            raise ValueError("Depth estimation requires a valid image frame")
        if not detections:
            return {}
        import cv2

        estimator = self._get_estimator()
        results: dict[int, str] = {}
        try:
            with tempfile.TemporaryDirectory(prefix="visionaid_depth_") as temporary_dir:
                image_path = Path(temporary_dir) / "frame.jpg"
                if not cv2.imwrite(str(image_path), frame.image):
                    raise OSError("Could not encode the frame for relative depth estimation")
                depth_map = estimator.estimate(image_path)
                self.load_time_ms = getattr(estimator, "load_time_ms", None)
                self.inference_latency_ms = getattr(
                    estimator, "last_inference_latency_ms", None
                )
                for detection in detections:
                    result = estimator.relative_depth_for_box(
                        depth_map, detection.bounding_box
                    )
                    if result.value is not None:
                        results[detection.track_id or -1] = result.category
            self.last_error = None
            return results
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            if not self.retain_model:
                self.estimator = None
                del estimator
                gc.collect()

    def close(self) -> None:
        estimator = self.estimator
        self.estimator = None
        del estimator
        gc.collect()
