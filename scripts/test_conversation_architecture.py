from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.conversation_engine import (
    ConversationEngine,
    DecisionKind,
    LLMDecision,
)
from src.llm.message import ConversationMemory
from src.llm.tool_interface import (
    CapabilityKind,
    CapabilityRegistry,
    CapabilityRequest,
    CapabilityResult,
    CapabilityProvider,
    GroundingData,
)


class MockProvider(CapabilityProvider):
    def __init__(self) -> None:
        self.requests: list[CapabilityRequest] = []

    def provide(self, request: CapabilityRequest) -> CapabilityResult:
        self.requests.append(request)
        if request.capability == "vision.object_search":
            target = request.arguments["target"]
            return CapabilityResult(
                available=True,
                data={"found": True, "target": target, "position": "left"},
                grounding=GroundingData(
                    target_information={"target": target, "position": "left"}
                ),
            )
        if request.capability == "navigation.guidance":
            return CapabilityResult(
                available=True,
                data={"target": request.arguments["target"], "guidance_ready": True},
                grounding=GroundingData(target_information={"target": request.arguments["target"]}),
            )
        if request.capability == "vision.scene_awareness":
            return CapabilityResult(
                available=True,
                data={"objects": ["desk", "bottle"]},
                grounding=GroundingData(scene_information={"objects": ["desk", "bottle"]}),
            )
        if request.capability == "safety.proximity":
            return CapabilityResult(
                available=True,
                data={"state": "clear", "too_close": False},
                grounding=GroundingData(safety_state={"state": "clear", "too_close": False}),
            )
        return CapabilityResult.unavailable("Mock capability is not configured")


class MockLLM:
    def decide(self, user_message: str, memory: ConversationMemory) -> LLMDecision:
        normalized = user_message.lower()
        if "bored" in normalized:
            return LLMDecision(
                DecisionKind.ANSWER,
                "Want to turn that boredom into a tiny adventure?",
                topic="casual conversation",
            )
        if "joke" in normalized:
            return LLMDecision(
                DecisionKind.ANSWER,
                "Why did the bicycle fall over? It was two-tired.",
                topic="humor",
            )
        if "looking for" in normalized and "bottle" in normalized:
            return LLMDecision(
                DecisionKind.INFORMATION,
                capability_request=CapabilityRequest(
                    "vision.object_search",
                    CapabilityKind.INFORMATION,
                    {"target": "bottle"},
                ),
                topic="finding an object",
            )
        if "take me to it" in normalized or "guide me to it" in normalized:
            return LLMDecision(
                DecisionKind.ACTION,
                capability_request=CapabilityRequest(
                    "navigation.guidance",
                    CapabilityKind.ACTION,
                    {"target": {"$context": "current_target"}},
                ),
                topic="guidance",
                clarification=(
                    "Which item would you like me to guide you to?"
                    if memory.current_target is None
                    else None
                ),
            )
        if "what do you see" in normalized:
            return LLMDecision(
                DecisionKind.INFORMATION,
                capability_request=CapabilityRequest(
                    "vision.scene_awareness", CapabilityKind.INFORMATION
                ),
                topic="scene awareness",
            )
        if "too close" in normalized:
            return LLMDecision(
                DecisionKind.INFORMATION,
                capability_request=CapabilityRequest(
                    "safety.proximity", CapabilityKind.INFORMATION
                ),
                topic="safety",
            )
        return LLMDecision(DecisionKind.ANSWER, "I’m listening.")

    def respond_to_capability(
        self,
        user_message: str,
        request: CapabilityRequest,
        result: CapabilityResult,
        memory: ConversationMemory,
    ) -> str:
        if not result.available:
            return "I can't access that information right now."
        if request.capability == "vision.object_search":
            if result.data.get("found"):
                return f"I found your {result.data['target']} on your {result.data['position']}."
            return "I don't see it right now."
            return LLMDecision(DecisionKind.ANSWER, "I'm listening.")
        if request.capability == "navigation.guidance":
            return f"I can guide you toward the {result.data['target']}."
        if request.capability == "vision.scene_awareness":
            objects = ", a ".join(result.data["objects"])
            return f"I can see a {objects}."
        if request.capability == "safety.proximity":
            if result.data["state"] == "clear":
                return "Nothing is flagged as too close right now."
            return "There is something close by."
        return "I received the result."


def make_engine() -> tuple[ConversationEngine, MockProvider]:
    provider = MockProvider()
    registry = CapabilityRegistry()
    for capability in (
        "vision.object_search",
        "navigation.guidance",
        "vision.scene_awareness",
        "safety.proximity",
    ):
        registry.register(capability, provider)
    return ConversationEngine(MockLLM(), registry), provider


def main() -> None:
    engine, provider = make_engine()
    normal = engine.process("Hey Friday, I'm bored.")
    assert "boredom" in normal and not provider.requests
    print("NORMAL CONVERSATION: PASS")

    search = engine.process("I'm looking for my bottle.")
    assert len(provider.requests) == 1
    assert provider.requests[-1].capability == "vision.object_search"
    assert "found your bottle" in search.lower()
    print("NATURAL TASK REQUEST: PASS")

    guidance = engine.process("Take me to it.")
    assert provider.requests[-1].capability == "navigation.guidance"
    assert provider.requests[-1].arguments["target"] == "bottle"
    assert "guide you toward the bottle" in guidance.lower()
    print("FOLLOW-UP REFERENT: PASS")

    scene = engine.process("What do you see?")
    assert provider.requests[-1].capability == "vision.scene_awareness"
    assert "desk" in scene and "bottle" in scene
    print("SCENE-AWARENESS REQUEST: PASS")

    safety = engine.process("Is anything too close?")
    assert provider.requests[-1].capability == "safety.proximity"
    assert "nothing is flagged as too close" in safety.lower()
    print("SAFETY REQUEST: PASS")

    joke_engine, joke_provider = make_engine()
    joke = joke_engine.process("Tell me a joke.")
    assert "two-tired" in joke and not joke_provider.requests
    print("JOKE WITHOUT CAPABILITY: PASS")

    empty_context_engine, empty_context_provider = make_engine()
    clarification = empty_context_engine.process("Take me to it.")
    assert "which item" in clarification.lower()
    assert not empty_context_provider.requests
    print("NO-CONTEXT CLARIFICATION: PASS")

    replies = [normal, search, guidance, scene, safety, joke, clarification]
    internal_markers = (
        "vision.object_search",
        "navigation.guidance",
        "vision.scene_awareness",
        "safety.proximity",
        "capabilityrequest",
    )
    assert all(
        marker not in reply.lower()
        for reply in replies
        for marker in internal_markers
    )
    print("INTERNAL REPRESENTATIONS HIDDEN: PASS")
    print("CONVERSATION ARCHITECTURE: PASS")


if __name__ == "__main__":
    main()