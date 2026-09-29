from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data/test_images/webcam_warm_frame.jpg"
FRAME_COUNT = 30
FRAME_INTERVAL_SECONDS = 0.08
DARK_BRIGHTNESS_THRESHOLD = 30.0


@dataclass
class FrameMeasurement:
    number: int
    valid: bool
    brightness: float | None = None
    min_pixel: int | None = None
    max_pixel: int | None = None
    resolution: str = "unavailable"
    frame: np.ndarray | None = None


def open_default_camera() -> tuple[cv2.VideoCapture | None, str]:
    errors: list[str] = []
    for backend, name in ((cv2.CAP_DSHOW, "CAP_DSHOW"), (cv2.CAP_ANY, "CAP_ANY")):
        camera = cv2.VideoCapture(0, backend)
        if camera.isOpened():
            return camera, name
        camera.release()
        errors.append(f"VideoCapture(0, {name}) returned isOpened()=False")
    return None, "; ".join(errors)


def measure_frame(number: int, ok: bool, frame: np.ndarray | None) -> FrameMeasurement:
    if not ok or frame is None or frame.ndim != 3:
        return FrameMeasurement(number=number, valid=False)
    height, width = frame.shape[:2]
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return FrameMeasurement(
        number=number,
        valid=True,
        brightness=float(np.mean(grayscale)),
        min_pixel=int(frame.min()),
        max_pixel=int(frame.max()),
        resolution=f"{width}x{height}",
        frame=frame.copy(),
    )


def main() -> None:
    camera, backend = open_default_camera()
    if camera is None:
        print("CAMERA = FAIL")
        print(f"CAMERA_CAPTURE_FAILED: {backend}")
        raise SystemExit(1)

    measurements: list[FrameMeasurement] = []
    try:
        print(f"CAMERA = PASS (backend={backend})")
        print("FRAME | VALID | BRIGHTNESS | MIN | MAX | RESOLUTION | UNUSUALLY_DARK")
        for number in range(1, FRAME_COUNT + 1):
            ok, frame = camera.read()
            measurement = measure_frame(number, ok, frame)
            measurements.append(measurement)
            dark = measurement.brightness is not None and measurement.brightness < DARK_BRIGHTNESS_THRESHOLD
            print(
                f"{number:>5} | {str(measurement.valid):<5} | "
                f"{measurement.brightness if measurement.brightness is not None else 'n/a':>10} | "
                f"{measurement.min_pixel if measurement.min_pixel is not None else 'n/a':>3} | "
                f"{measurement.max_pixel if measurement.max_pixel is not None else 'n/a':>3} | "
                f"{measurement.resolution:<10} | {dark}"
            )
            if number < FRAME_COUNT:
                time.sleep(FRAME_INTERVAL_SECONDS)
    finally:
        camera.release()

    valid_frames = [measurement for measurement in measurements if measurement.valid]
    warm_frame = next(
        (
            measurement for measurement in valid_frames
            if measurement.brightness is not None and measurement.brightness >= DARK_BRIGHTNESS_THRESHOLD
        ),
        None,
    )
    print(f"FRAMES CAPTURED: {len(valid_frames)}")
    print(f"FIRST FRAME BRIGHTNESS: {measurements[0].brightness if measurements else 'n/a'}")
    print(f"LAST FRAME BRIGHTNESS: {measurements[-1].brightness if measurements else 'n/a'}")
    if warm_frame is None:
        print("WARM_FRAME_FOUND = FALSE")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT_PATH), warm_frame.frame):
        print("WARM_FRAME_FOUND = FALSE")
        print(f"SAVE_FAILED: could not save {OUTPUT_PATH}")
        raise SystemExit(1)
    print("WARM_FRAME_FOUND = TRUE")
    print(f"SELECTED FRAME NUMBER: {warm_frame.number}")
    print(f"SELECTED FRAME BRIGHTNESS: {warm_frame.brightness:.2f}")
    print(f"SELECTED FRAME RESOLUTION: {warm_frame.resolution}")
    print(f"SAVED IMAGE: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
