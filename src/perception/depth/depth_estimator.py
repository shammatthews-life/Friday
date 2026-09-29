from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


@dataclass(frozen=True)
class RelativeDepthResult:
    """A scene-relative depth estimate, never a calibrated physical distance."""

    value: float | None
    category: str


class DepthEstimator:
    """Local, CPU-only runner for the staged Depth Anything V2 Small checkpoint.

    The category thresholds are heuristic: the object's robust regional median is
    compared with the 33rd and 67th percentiles of the current image's relative
    depth map.  They describe only relative ordering within that image.
    """

    def __init__(
        self,
        model_dir: str | Path = "models/depth/depth-anything-v2-small",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.processor = None
        self.model = None
        self.load_time_ms: float | None = None
        self.last_inference_latency_ms: float | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        required_files = ("config.json", "preprocessor_config.json", "model.safetensors")
        missing = [name for name in required_files if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Local depth checkpoint is incomplete: {', '.join(missing)}")

        # Enforce local-only Hugging Face loading for this process.
        os.environ["HF_HUB_OFFLINE"] = "1"
        start = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.model_dir, local_files_only=True
        ).to("cpu")
        self.model.eval()
        self.load_time_ms = (time.perf_counter() - start) * 1000

    def estimate(self, image_path: str | Path) -> np.ndarray:
        """Return the model's relative-depth map at the source image resolution."""
        self.load()
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        self.last_inference_latency_ms = (time.perf_counter() - start) * 1000
        processed = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )
        return processed[0]["predicted_depth"].detach().cpu().numpy()

    def relative_depth_for_box(
        self, depth_map: np.ndarray, bounding_box: tuple[float, float, float, float]
    ) -> RelativeDepthResult:
        """Use an inset bounding-box region and its median, not a single pixel."""
        if depth_map.ndim != 2 or not np.isfinite(depth_map).any():
            return RelativeDepthResult(None, "unable to estimate")

        height, width = depth_map.shape
        left, top, right, bottom = bounding_box
        x1, x2 = sorted((max(0, int(np.floor(left))), min(width, int(np.ceil(right)))))
        y1, y2 = sorted((max(0, int(np.floor(top))), min(height, int(np.ceil(bottom)))))
        if x2 <= x1 or y2 <= y1:
            return RelativeDepthResult(None, "unable to estimate")

        # Exclude a 20% border where detector-box/background bleed is most likely.
        inset_x = max(1, int((x2 - x1) * 0.2))
        inset_y = max(1, int((y2 - y1) * 0.2))
        inner_x1, inner_x2 = x1 + inset_x, x2 - inset_x
        inner_y1, inner_y2 = y1 + inset_y, y2 - inset_y
        region = depth_map[
            inner_y1:inner_y2, inner_x1:inner_x2
        ] if inner_x2 > inner_x1 and inner_y2 > inner_y1 else depth_map[y1:y2, x1:x2]
        values = region[np.isfinite(region)]
        if values.size < 9:
            return RelativeDepthResult(None, "unable to estimate")

        value = float(np.median(values))
        scene_values = depth_map[np.isfinite(depth_map)]
        far_threshold, near_threshold = np.percentile(scene_values, [33, 67])
        # Depth Anything V2's relative output is treated as inverse-depth ordering:
        # higher values mean relatively closer, not a metric distance.
        if value >= near_threshold:
            category = "relatively near"
        elif value <= far_threshold:
            category = "relatively far"
        else:
            category = "relatively middle-distance"
        return RelativeDepthResult(value, category)
