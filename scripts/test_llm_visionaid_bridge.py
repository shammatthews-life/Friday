from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.scene_memory import SceneMemory
from src.core.scene_state import SceneObject, SceneState
from src.llm.conversation_engine import ConversationEngine, DecisionKind, LLMDecision
from src.llm.message import ConversationMemory
from src.llm.tool_interface import (
    CapabilityKind,
    CapabilityRegistry,
    CapabilityRequest,
    CapabilityResult,
)
from src.llm.visionaid_bridge import (
    OBJECT_SEARCH,
    RELATIVE_DEPTH,
    SCENE_AWARENESS,
    VisionAidBridge,
)
from src.search.object_search import ObjectSearch, TargetState


class MockDepthProvider:
    def __init__(self) -> None:
        self.requested_labels: list[str] = []

    def __call__(self, scene_object: SceneObject, image_path: str | Path | None) -> dict[str, Any]:
        self.requested_labels.append(scene_object.label)
        return {"category": "relatively near", "available": True}


class RecordingBridge(VisionAidBridge):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requests: list[CapabilityRequest] = []

    def provide(self, request: CapabilityRequest) -> CapabilityResult:
        self.requests.append(request)
        return super().provide(request)


class MockLLM:
    def decide(self, user_message: str, memory: ConversationMemory) -> LLMDecision:
        normalized = user_message.lower()
        if "what do you see" in normalized or "what's around" in normalized:
            return LLMDecision(
                DecisionKind.INFORMATION,
                capability_request=CapabilityRequest(
                    SCENE_AWARENESS, CapabilityKind.INFORMATION
                ),
                topic="the current scene",
            )
        if "what about the bottle" in normalized:
            return LLMDecision(
                DecisionKind.INFORMATION,
                capability_request=CapabilityRequest(
                    SCENE_AWARENESS,
                    CapabilityKind.INFORMATION,
                    {"target": "bottle"},
                ),
                topic="the bottle",
                current_referent="bottle",
            )
        if "how far" in normalized or "near" in normalized:
            return LLMDecision(
                DecisionKind.INFORMATION,
                capability_request=CapabilityRequest(
                    RELATIVE_DEPTH,
                    CapabilityKind.INFORMATION,
                    {"target": {"$context": "current_referent"}},
                ),
                topic="relative depth",
            )
        if "looking for a chair" in normalized:
            return LLMDecision(
                DecisionKind.ANSWER,
                "I can help you look for a chair.",
                topic="the chair",
                current_referent="chair",
            )
        if "can you find it" in normalized:
            return LLMDecision(
                DecisionKind.INFORMATION,
                capability_request=CapabilityRequest(
                    OBJECT_SEARCH,
                    CapabilityKind.INFORMATION,
                    {"target": {"$context": "current_referent"}},
                ),
                topic="finding the chair",
            )
        if "joke" in normalized:
            return LLMDecision(
                DecisionKind.ANSWER,
                "Why did the bicycle fall over? It was two-tired.",
                topic="humor",
            )
        return LLMDecision(DecisionKind.ANSWER, "I'm listening.")

    def respond_to_capability(
        self,
        user_message: str,
        request: CapabilityRequest,
        result: CapabilityResult,
        memory: ConversationMemory,
    ) -> str:
        if not result.available or not result.data.get("scene_available", True):
            return "I can't see the space right now."
        if request.capability == SCENE_AWARENESS:
            target_object = result.data.get("target_object")
            if target_object:
                location = (
                    "in the center"
                    if target_object["position"] == "center"
                    else f"on your {target_object['position']}"
                )
                return (
                    f"The {target_object['label']} is {location}."
                )
            objects = result.data["objects"]
            descriptions = [
                (
                    f"a {obj['label']} in the center"
                    if obj["position"] == "center"
                    else f"a {obj['label']} on your {obj['position']}"
                )
                for obj in objects
            ]
            if len(descriptions) == 3:
                listing = f"{descriptions[0]}, {descriptions[1]}, and {descriptions[2]}"
            else:
                listing = ", ".join(descriptions)
            return f"I can see {listing}." if descriptions else "I don't see anything clearly right now."
        if request.capability == RELATIVE_DEPTH:
            if not result.data.get("object_found"):
                return "I don't see that object right now."
            if not result.data.get("depth_available"):
                return f"I can see the {result.data['target']}, but can't estimate its relative depth."
            return (
                f"The {result.data['target']} appears "
                f"{result.data['category']}."
            )
        if request.capability == OBJECT_SEARCH:
            if not result.data.get("found"):
                return f"I don't see a {result.data['target']} right now."
            position = result.data.get("position")
            return f"I found the {result.data['target']} in the {position}."
        return "I can't access that information right now."


def make_engine() -> tuple[ConversationEngine, RecordingBridge, MockDepthProvider]:
    scene = SceneState(
        objects=[
            SceneObject("person", 0.91, position_category="right"),
            SceneObject("bottle", 0.84, position_category="left"),
            SceneObject("chair", 0.88, position_category="center"),
        ]
    )
    memory = SceneMemory()
    memory.update(scene)
    depth_provider = MockDepthProvider()
    bridge = RecordingBridge(
        scene_provider=lambda: scene,
        scene_memory=memory,
        object_search=ObjectSearch(),
        depth_provider=depth_provider,
    )
    registry = bridge.register(CapabilityRegistry())
    return ConversationEngine(MockLLM(), registry), bridge, depth_provider


def main() -> None:
    engine, bridge, depth_provider = make_engine()
    replies = [
        engine.process("Hey Friday, what do you see?"),
        engine.process("What about the bottle?"),
        engine.process("How far is it?"),
        engine.process("I think I'm looking for a chair."),
        engine.process("Can you find it?"),
    ]
    joke = engine.process("Tell me a joke.")
    replies.append(joke)

    assert "person" in replies[0] and "bottle" in replies[0] and "chair" in replies[0]
    assert "right" in replies[0] and "left" in replies[0] and "center" in replies[0]
    assert "bottle" in replies[1].lower() and "left" in replies[1].lower()
    assert "relatively near" in replies[2].lower()
    assert "chair" in replies[3].lower()
    assert "found the chair" in replies[4].lower() and "center" in replies[4].lower()
    assert "two-tired" in joke

    assert [request.capability for request in bridge.requests] == [
        SCENE_AWARENESS,
        SCENE_AWARENESS,
        RELATIVE_DEPTH,
        OBJECT_SEARCH,
    ]
    assert bridge.requests[2].arguments["target"] == "bottle"
    assert bridge.requests[3].arguments["target"] == "chair"
    assert depth_provider.requested_labels == ["bottle"]
    assert bridge.object_search is not None
    assert bridge.object_search.target is not None
    assert bridge.object_search.target.state is TargetState.FOUND
    assert bridge.object_search.target.label == "chair"
    assert engine.memory.current_referent == "chair"
    assert engine.memory.current_target == "chair"

    internal_markers = (
        SCENE_AWARENESS,
        OBJECT_SEARCH,
        RELATIVE_DEPTH,
        "capabilityrequest",
        "scene_available",
        "depth_available",
        "target_information",
        "relative_depth",
    )
    assert all(
        marker not in reply.lower()
        for reply in replies
        for marker in internal_markers
    )
    assert all("{" not in reply and "}" not in reply for reply in replies)
    assert len(bridge.requests) == 4
    assert "bicycle" in joke.lower() and len(bridge.requests) == 4

    heavy_modules = {"torch", "ultralytics", "transformers", "piper"}
    assert not (heavy_modules & sys.modules.keys())

    print("SCENE AWARENESS: PASS - " + replies[0])
    print("BOTTLE FOLLOW-UP: PASS - " + replies[1])
    print("RELATIVE DEPTH: PASS - " + replies[2])
    print("CONTEXT RESOLUTION: PASS - 'it' resolved to bottle; 'it' later resolved to chair")
    print("OBJECT SEARCH: PASS - " + replies[4])
    print("NORMAL CONVERSATION WITHOUT CAPABILITY: PASS - " + joke)
    print("INTERNAL CAPABILITY IDENTIFIERS HIDDEN: PASS")
    print("NO HEAVY MODEL MODULES LOADED: PASS")
    print("LLM VISIONAID BRIDGE: PASS")


if __name__ == "__main__":
    main()
