from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/detection/yolo26n.pt"
OUTPUT_PATH = ROOT / "data/test_images/webcam_detection_best.jpg"
FRAME_COUNT = 5


def open_default_camera() -> tuple[cv2.VideoCapture | None, str]:
    errors: list[str] = []
    for backend, name in ((cv2.CAP_DSHOW, "CAP_DSHOW"), (cv2.CAP_ANY, "CAP_ANY")):
        camera = cv2.VideoCapture(0, backend)
        if camera.isOpened():
            return camera, name
        camera.release()
        errors.append(f"VideoCapture(0, {name}) returned isOpened()=False")
    return None, "; ".join(errors)


def position_for_box(left: float, right: float, frame_width: int) -> str:
    normalized_x = ((left + right) / 2) / frame_width if frame_width else 0.5
    if normalized_x < 1 / 3:
        return "left"
    if normalized_x > 2 / 3:
        return "right"
    return "center"


def labels_for_result(result, frame_width: int) -> list[str]:
    descriptions: list[str] = []
    for box, confidence, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        left, _, right, _ = [float(value) for value in box]
        label = str(result.names[int(class_id)])
        descriptions.append(
            f"{label} ({float(confidence):.3f}, {position_for_box(left, right, frame_width)})"
        )
    return descriptions


def main() -> None:
    camera, backend = open_default_camera()
    if camera is None:
        print("CAMERA = FAIL")
        print(f"CAMERA_CAPTURE_FAILED: {backend}")
        raise SystemExit(1)

    try:
        print(f"CAMERA = PASS (backend={backend})")
        time.sleep(1.0)

        load_start = time.perf_counter()
        model = YOLO(str(MODEL_PATH))
        model_load_ms = (time.perf_counter() - load_start) * 1000
        print(f"MODEL LOAD TIME MS: {model_load_ms:.2f}")

        warmup_ok, warmup_frame = camera.read()
        if not warmup_ok or warmup_frame is None:
            print("REAL_SCENE_DETECTION = NOT_CONFIRMED")
            print("CAMERA_CAPTURE_FAILED: warm-up VideoCapture.read() returned no valid frame")
            raise SystemExit(1)
        model.predict(source=warmup_frame, device="cpu", verbose=False, imgsz=640)
        print("WARM-UP = PASS")

        latencies_ms: list[float] = []
        labels_seen: set[str] = set()
        frames_with_detections = 0
        best_result = None
        best_detection_count = -1

        for frame_number in range(1, FRAME_COUNT + 1):
            ok, frame = camera.read()
            if not ok or frame is None:
                print(f"FRAME {frame_number}: CAMERA_CAPTURE_FAILED")
                continue
            height, width = frame.shape[:2]
            inference_start = time.perf_counter()
            result = model.predict(source=frame, device="cpu", verbose=False, imgsz=640)[0]
            latency_ms = (time.perf_counter() - inference_start) * 1000
            latencies_ms.append(latency_ms)
            descriptions = labels_for_result(result, width)
            detection_count = len(result.boxes)
            labels_seen.update(str(result.names[int(class_id)]) for class_id in result.boxes.cls)
            if detection_count:
                frames_with_detections += 1
            if detection_count > best_detection_count:
                best_detection_count = detection_count
                best_result = result
            print(f"FRAME {frame_number}: resolution={width}x{height}; detections={detection_count}; "
                  f"labels={', '.join(descriptions) if descriptions else 'none'}; "
                  f"warm_inference_latency_ms={latency_ms:.2f}")

        if best_result is not None:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(OUTPUT_PATH), best_result.plot()):
                print(f"SAVE_FAILED: could not save annotated frame to {OUTPUT_PATH}")
                raise SystemExit(1)

        average_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        average_fps = 1000 / average_latency_ms if average_latency_ms else 0.0
        print(f"TOTAL FRAMES TESTED: {len(latencies_ms)}")
        print(f"FRAMES WITH >=1 DETECTION: {frames_with_detections}")
        print(f"AVERAGE WARM INFERENCE LATENCY MS: {average_latency_ms:.2f}")
        print(f"AVERAGE FPS: {average_fps:.2f}")
        print(f"HIGHEST DETECTION COUNT: {max(best_detection_count, 0)}")
        print(f"LABELS ACROSS FIVE FRAMES: {', '.join(sorted(labels_seen)) if labels_seen else 'none'}")
        print(f"ANNOTATED BEST IMAGE: {OUTPUT_PATH}")
        if frames_with_detections:
            print("REAL_SCENE_DETECTION_CONFIRMED")
        else:
            print("NO_REAL_SCENE_DETECTION_CONFIRMED")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
