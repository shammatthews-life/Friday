from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.friday import FridayAssistant


def main() -> None:
    image_path = ROOT / "data/test_images/scene_common_objects.png"
    assistant = FridayAssistant(ROOT / "models/detection/yolo26n.pt")
    scene = assistant.detect(image_path)
    if not scene.objects:
        print("TEST: FAIL - YOLO26n detected no target object; depth was not run.")
        raise SystemExit(1)

    target = scene.objects[0]
    bounding_box = assistant.bounding_box_for(target)
    if bounding_box is None:
        print("TEST: FAIL - no bounding box is available for the detected target.")
        raise SystemExit(1)

    result = assistant.relative_depth_for_object(image_path, target)
    estimator = assistant.depth_estimator
    response = assistant.respond(f"How far is the {target.label}?", scene, image_path)

    print(f"DETECTED OBJECT: {target.label}")
    print(f"CONFIDENCE: {target.confidence:.3f}")
    print(f"BOUNDING BOX: {bounding_box}")
    print(f"RELATIVE-DEPTH VALUE: {result.value}")
    print(f"RELATIVE-DEPTH CATEGORY: {result.category}")
    print(f"FRIDAY RESPONSE: {response}")
    print(f"DEPTH LOAD TIME MS: {estimator.load_time_ms if estimator else None}")
    print(f"DEPTH LATENCY MS: {estimator.last_inference_latency_ms if estimator else None}")
    print("NOTE: This is a synthetic pipeline test, not an accuracy or metric-distance test.")
    print("TEST: PASS" if result.value is not None else "TEST: FAIL")
    if result.value is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
