from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.types import PerceptionFrame


@dataclass(frozen=True)
class FrameValidation:
    valid: bool
    brightness: float | None
    contrast: float | None
    reason: str | None = None


def validate_frame(
    frame: PerceptionFrame,
    *,
    minimum_brightness: float = 8.0,
    minimum_contrast: float = 1.0,
) -> FrameValidation:
    image = frame.image
    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
        return FrameValidation(False, None, None, "empty or invalid image array")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        return FrameValidation(False, None, None, "zero-sized frame")
    if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
        return FrameValidation(False, None, None, "unsupported channel count")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        return FrameValidation(False, None, None, "frame contains invalid numeric data")
    values = image.astype(np.float32, copy=False)
    if values.ndim == 3 and values.shape[2] >= 3:
        gray = values[:, :, :3].mean(axis=2)
    elif values.ndim == 3:
        gray = values[:, :, 0]
    else:
        gray = values
    brightness = float(gray.mean())
    contrast = float(gray.std())
    if brightness < minimum_brightness:
        return FrameValidation(False, brightness, contrast, "frame is too dark")
    if contrast < minimum_contrast:
        return FrameValidation(False, brightness, contrast, "frame is blank or nearly uniform")
    return FrameValidation(True, brightness, contrast)
