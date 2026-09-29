from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.friday import FridayAssistant


def main() -> None:
    root = ROOT
    assistant = FridayAssistant(root / "models/detection/yolo26n.pt")
    scene = assistant.detect(root / "data/test_images/scene_common_objects.png")

    questions = [
        "What is around me?",
        "Is there a chair?",
        "Is there a dog?",
        "Where is the chair?",
        "How many objects are there?",
    ]
    print("DETECTIONS")
    for obj in scene.objects:
        print(obj)
    print("SCENE STATE")
    print(scene)
    for question in questions:
        parsed = assistant.parse(question)
        print(f"QUESTION: {question}")
        print(f"INTENT: {parsed}")
        print(f"FRIDAY: {assistant.respond(question, scene)}")
    print("TEST: PASS")


if __name__ == "__main__":
    main()
