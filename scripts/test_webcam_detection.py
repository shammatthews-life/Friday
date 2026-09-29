from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/detection/yolo26n.pt"
OUTPUT_PATH = ROOT / "data/test_images/webcam_detection_test.jpg"


def open_default_camera() -> tuple[cv2.VideoCapture | None, str]:
    attempts = ((cv2.CAP_DSHOW, "CAP_DSHOW"), (cv2.CAP_ANY, "CAP_ANY"))
    errors: list[str] = []
    for backend, name in attempts:
        camera = cv2.VideoCapture(0, backend)
        if camera.isOpened():
            return camera, name
        camera.release()
        errors.append(f"VideoCapture(0, {name}) returned isOpened()=False")
    return None, "; ".join(errors)


def position_for_box(left: float, right: float, frame_width: int) -> str:
    center_x = (left + right) / 2
    normalized_x = center_x / frame_width if frame_width else 0.5
    if normalized_x < 1 / 3:
        return "left"
    if normalized_x > 2 / 3:
        return "right"
    return "center"


def main() -> None:
    camera, backend = open_default_camera()
    if camera is None:
        print("CAMERA = FAIL")
        print("FRAME_CAPTURE = FAIL")
        print("YOLO = FAIL")
        print(f"CAMERA_CAPTURE_FAILED: {backend}")
        raise SystemExit(1)

    try:
        ok, frame = camera.read()
    except cv2.error as error:
        print("CAMERA = PASS")
        print("FRAME_CAPTURE = FAIL")
        print("YOLO = FAIL")
        print(f"CAMERA_CAPTURE_FAILED: backend={backend}; OpenCV error: {error}")
        raise SystemExit(1)
    finally:
        camera.release()

    if not ok or frame is None:
        print("CAMERA = PASS")
        print("FRAME_CAPTURE = FAIL")
        print("YOLO = FAIL")
        print(f"CAMERA_CAPTURE_FAILED: backend={backend}; VideoCapture.read() returned ok={ok}, frame={frame is not None}")
        raise SystemExit(1)

    height, width = frame.shape[:2]
    print("CAMERA = PASS")
    print("FRAME_CAPTURE = PASS")
    print(f"FRAME RESOLUTION: {width}x{height}")

    try:
        model = YOLO(str(MODEL_PATH))
        start = time.perf_counter()
        result = model.predict(source=frame, device="cpu", verbose=False, imgsz=640)[0]
        latency_ms = (time.perf_counter() - start) * 1000
    except Exception as error:
        print("YOLO = FAIL")
        print(f"YOLO_FAILURE: {type(error).__name__}: {error}")
        raise SystemExit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT_PATH), result.plot()):
        print("YOLO = FAIL")
        print(f"YOLO_FAILURE: could not save annotated frame to {OUTPUT_PATH}")
        raise SystemExit(1)

    print("YOLO = PASS")
    print(f"INFERENCE LATENCY MS: {latency_ms:.2f}")
    print(f"NUMBER OF DETECTIONS: {len(result.boxes)}")
    if not len(result.boxes):
        print("NO_DETECTION")
    else:
        for box, confidence, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            left, top, right, bottom = [float(value) for value in box]
            label = str(result.names[int(class_id)])
            position = position_for_box(left, right, width)
            print(
                f"DETECTION: label={label}; confidence={float(confidence):.3f}; "
                f"bounding_box=({left:.1f}, {top:.1f}, {right:.1f}, {bottom:.1f}); "
                f"position={position}"
            )
    print(f"ANNOTATED IMAGE: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
