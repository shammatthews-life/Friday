from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/detection/yolo26n.pt"
WEBCAM_IMAGE = ROOT / "data/test_images/webcam_detection_best.jpg"
SYNTHETIC_IMAGE = ROOT / "data/test_images/scene_common_objects.png"
DEFAULT_OUTPUT = ROOT / "data/test_images/webcam_diagnostic_default.jpg"
LOWCONF_OUTPUT = ROOT / "data/test_images/webcam_diagnostic_lowconf.jpg"


def print_detections(name: str, result) -> int:
    count = len(result.boxes)
    print(f"{name} DETECTION COUNT: {count}")
    if not count:
        print(f"{name} LABELS: none")
        return 0
    for box, confidence, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        left, top, right, bottom = [float(value) for value in box]
        label = str(result.names[int(class_id)])
        print(
            f"{name} DETECTION: label={label}; confidence={float(confidence):.3f}; "
            f"bounding_box=({left:.1f}, {top:.1f}, {right:.1f}, {bottom:.1f})"
        )
    return count


def main() -> None:
    webcam = cv2.imread(str(WEBCAM_IMAGE), cv2.IMREAD_COLOR)
    if webcam is None:
        print(f"FAILURE: could not load saved webcam image: {WEBCAM_IMAGE}")
        raise SystemExit(1)

    height, width, channels = webcam.shape
    grayscale = cv2.cvtColor(webcam, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(grayscale))
    unusually_dark = mean_brightness < 30.0
    overexposed = mean_brightness > 225.0
    print(f"IMAGE WIDTH: {width}")
    print(f"IMAGE HEIGHT: {height}")
    print(f"IMAGE CHANNELS: {channels}")
    print(f"MIN PIXEL VALUE: {int(webcam.min())}")
    print(f"MAX PIXEL VALUE: {int(webcam.max())}")
    print(f"MEAN PIXEL VALUE: {float(np.mean(webcam)):.2f}")
    print(f"MEAN BRIGHTNESS: {mean_brightness:.2f}")
    print(f"UNUSUALLY DARK: {unusually_dark} (heuristic: mean brightness < 30)")
    print(f"OVEREXPOSED: {overexposed} (heuristic: mean brightness > 225)")

    model = YOLO(str(MODEL_PATH))
    webcam_default = model.predict(source=str(WEBCAM_IMAGE), device="cpu", verbose=False, imgsz=640)[0]
    webcam_lowconf = model.predict(
        source=str(WEBCAM_IMAGE), device="cpu", verbose=False, imgsz=640, conf=0.05
    )[0]
    synthetic_default = model.predict(source=str(SYNTHETIC_IMAGE), device="cpu", verbose=False, imgsz=640)[0]

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(DEFAULT_OUTPUT), webcam_default.plot()):
        print(f"FAILURE: could not save {DEFAULT_OUTPUT}")
        raise SystemExit(1)
    if not cv2.imwrite(str(LOWCONF_OUTPUT), webcam_lowconf.plot()):
        print(f"FAILURE: could not save {LOWCONF_OUTPUT}")
        raise SystemExit(1)

    webcam_default_count = print_detections("WEBCAM DEFAULT", webcam_default)
    webcam_lowconf_count = print_detections("WEBCAM LOW-CONFIDENCE DIAGNOSTIC", webcam_lowconf)
    synthetic_count = print_detections("SYNTHETIC DEFAULT", synthetic_default)
    print(f"SAVED DEFAULT DIAGNOSTIC: {DEFAULT_OUTPUT}")
    print(f"SAVED LOW-CONFIDENCE DIAGNOSTIC: {LOWCONF_OUTPUT}")

    if synthetic_count and (unusually_dark or overexposed):
        conclusion = "CAMERA_FRAME_PROBLEM"
    elif synthetic_count and not webcam_default_count and webcam_lowconf_count:
        conclusion = "NORMAL_CONFIDENCE_TOO_HIGH"
    elif synthetic_count and not webcam_lowconf_count:
        conclusion = "MODEL_SEES_NO_OBJECTS"
    else:
        conclusion = "OTHER / INCONCLUSIVE"
    print("COMPARISON:")
    print(f"A. SYNTHETIC IMAGE DETECTION: {synthetic_count}")
    print(f"B. WEBCAM DEFAULT-CONFIDENCE DETECTION: {webcam_default_count}")
    print(f"C. WEBCAM LOW-CONFIDENCE DETECTION: {webcam_lowconf_count}")
    print(f"CONCLUSION: {conclusion}")


if __name__ == "__main__":
    main()
