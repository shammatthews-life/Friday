from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/detection/yolo26n.pt"
WARM_IMAGE_PATH = ROOT / "data/test_images/webcam_warm_frame.jpg"
WARM_OUTPUT_PATH = ROOT / "data/test_images/webcam_warm_yolo.jpg"
FRESH_OUTPUT_PATH = ROOT / "data/test_images/webcam_fresh_yolo.jpg"
DARK_BRIGHTNESS_THRESHOLD = 30.0


def brightness(image: np.ndarray) -> float:
    return float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))


def position_for_box(left: float, right: float, width: int) -> str:
    normalized_x = ((left + right) / 2) / width if width else 0.5
    if normalized_x < 1 / 3:
        return "left"
    if normalized_x > 2 / 3:
        return "right"
    return "center"


def print_detections(name: str, result, width: int, latency_ms: float) -> None:
    print(f"{name} INFERENCE LATENCY MS: {latency_ms:.2f}")
    print(f"{name} DETECTION COUNT: {len(result.boxes)}")
    if not len(result.boxes):
        print(f"{name} DETECTIONS: none")
        return
    for box, confidence, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        left, top, right, bottom = [float(value) for value in box]
        print(
            f"{name} DETECTION: label={result.names[int(class_id)]}; "
            f"confidence={float(confidence):.3f}; "
            f"bounding_box=({left:.1f}, {top:.1f}, {right:.1f}, {bottom:.1f}); "
            f"position={position_for_box(left, right, width)}"
        )


def infer(model: YOLO, image: np.ndarray):
    start = time.perf_counter()
    result = model.predict(source=image, device="cpu", verbose=False, imgsz=640)[0]
    return result, (time.perf_counter() - start) * 1000


def main() -> None:
    warm_image = cv2.imread(str(WARM_IMAGE_PATH), cv2.IMREAD_COLOR)
    if warm_image is None:
        print("WARM_IMAGE_YOLO = FAIL")
        print(f"WARM_IMAGE_FAILURE: could not load {WARM_IMAGE_PATH}")
        raise SystemExit(1)
    warm_height, warm_width = warm_image.shape[:2]
    warm_brightness = brightness(warm_image)
    print(f"WARM IMAGE RESOLUTION: {warm_width}x{warm_height}")
    print(f"WARM IMAGE MEAN BRIGHTNESS: {warm_brightness:.2f}")

    model = YOLO(str(MODEL_PATH))
    try:
        warm_result, warm_latency_ms = infer(model, warm_image)
        WARM_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(WARM_OUTPUT_PATH), warm_result.plot()):
            raise RuntimeError(f"could not save {WARM_OUTPUT_PATH}")
        print("WARM_IMAGE_YOLO = PASS")
        print_detections("WARM IMAGE", warm_result, warm_width, warm_latency_ms)
        print(f"WARM IMAGE OUTPUT: {WARM_OUTPUT_PATH}")
    except Exception as error:
        print("WARM_IMAGE_YOLO = FAIL")
        print(f"WARM_IMAGE_FAILURE: {type(error).__name__}: {error}")
        raise SystemExit(1)

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        print("FRESH_CAMERA_YOLO = FAIL")
        print("CAMERA_FAILURE: VideoCapture(0, CAP_DSHOW) returned isOpened()=False")
        return
    try:
        print("CAMERA = PASS (backend=CAP_DSHOW)")
        time.sleep(1.0)
        ok, fresh_image = camera.read()
    except cv2.error as error:
        print("FRESH_CAMERA_YOLO = FAIL")
        print(f"CAMERA_FAILURE: OpenCV error: {error}")
        return
    finally:
        camera.release()

    if not ok or fresh_image is None:
        print("FRESH_CAMERA_YOLO = FAIL")
        print(f"CAMERA_FAILURE: VideoCapture.read() returned ok={ok}, frame={fresh_image is not None}")
        return

    fresh_height, fresh_width = fresh_image.shape[:2]
    fresh_brightness = brightness(fresh_image)
    brightness_normal = fresh_brightness >= DARK_BRIGHTNESS_THRESHOLD
    print(f"FRESH IMAGE RESOLUTION: {fresh_width}x{fresh_height}")
    print(f"FRESH IMAGE MEAN BRIGHTNESS: {fresh_brightness:.2f}")
    print(f"FRESH IMAGE NORMAL BRIGHTNESS: {brightness_normal}")
    try:
        fresh_result, fresh_latency_ms = infer(model, fresh_image)
        if not cv2.imwrite(str(FRESH_OUTPUT_PATH), fresh_result.plot()):
            raise RuntimeError(f"could not save {FRESH_OUTPUT_PATH}")
        print(f"FRESH_CAMERA_YOLO = {'PASS' if brightness_normal else 'FAIL'}")
        print_detections("FRESH IMAGE", fresh_result, fresh_width, fresh_latency_ms)
        print(f"FRESH IMAGE OUTPUT: {FRESH_OUTPUT_PATH}")
    except Exception as error:
        print("FRESH_CAMERA_YOLO = FAIL")
        print(f"FRESH_CAMERA_FAILURE: {type(error).__name__}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
