from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from test_video_context import make_complete_result
from src.llm.conversation_engine import ConversationEngine, DecisionKind, LLMDecision
from src.llm.message import ConversationMemory
from src.llm.tool_interface import CapabilityRequest, CapabilityResult
from src.llm.video_context_bridge import (
    VIDEO_QUESTION_CAPABILITY,
    VideoConversationBridge,
)
from src.perception.video.knowledge import VideoKnowledge
from src.perception.video.question import (
    VideoQuestionInterface,
    VideoQuestionKind,
    VideoQuestionStatus,
)


class MockQwen:
    def __init__(self) -> None:
        self.decisions = 0
        self.responses: list[tuple[str, CapabilityRequest, CapabilityResult]] = []

    def decide(self, user_message: str, memory: ConversationMemory) -> LLMDecision:
        self.decisions += 1
        return LLMDecision(DecisionKind.ANSWER, "Ordinary conversation.")

    def respond_to_capability(
        self,
        user_message: str,
        request: CapabilityRequest,
        result: CapabilityResult,
        memory: ConversationMemory,
    ) -> str:
        self.responses.append((user_message, request, result))
        if result.data.get("evidence_status") != "grounded":
            return "I don't have enough grounded video evidence to answer that."
        if user_message.casefold().startswith("what text"):
            entries = result.data["context"]["text_observations"]
            return "The video text includes " + ", ".join(item["text"] for item in entries) + "."
        if user_message.casefold().startswith("when did"):
            return "The grounded events show when the object appeared."
        return "The grounded video context is available."


def engine_for(
    knowledge: VideoKnowledge | None,
) -> tuple[ConversationEngine, MockQwen, VideoConversationBridge]:
    qwen = MockQwen()
    bridge = VideoConversationBridge(knowledge)
    return ConversationEngine(qwen, video_bridge=bridge), qwen, bridge


def test_question_classification_and_context_selection() -> None:
    knowledge = make_complete_result().knowledge
    interface = VideoQuestionInterface()
    full = interface.query("What happened in this video?", knowledge)
    assert full.kind is VideoQuestionKind.FULL_VIDEO
    assert full.status is VideoQuestionStatus.GROUNDED

    visual = interface.query("What objects appeared?", knowledge)
    assert visual.kind is VideoQuestionKind.VISUAL_EVENTS
    assert visual.context["text_observations"] == []
    assert visual.context["visual_events"]

    object_query = interface.query("When did the person appear?", knowledge)
    assert object_query.kind is VideoQuestionKind.OBJECT
    assert object_query.referent_track_id is not None

    text = interface.query("What text was shown?", knowledge)
    assert text.kind is VideoQuestionKind.TEXT
    assert text.context["visual_events"] == []
    assert text.context["text_observations"]
    exit_text = interface.query("When did EXIT appear?", knowledge)
    assert exit_text.kind is VideoQuestionKind.TEXT
    assert all(item["text"] == "EXIT" for item in exit_text.context["text_observations"])

    time = interface.query("What happened around 5 seconds?", knowledge)
    assert time.kind is VideoQuestionKind.TIME_RANGE
    assert time.context["time_ranges"]["requested"] == {
        "start_timestamp": 4.0,
        "end_timestamp": 6.0,
    }

    unsupported = interface.query(
        "What color was the person's shirt in this video?",
        knowledge,
    )
    assert unsupported.status is VideoQuestionStatus.UNSUPPORTED
    assert unsupported.context is None
    unsupported_text_meaning = interface.query(
        "What does the EXIT sign mean?",
        knowledge,
    )
    assert unsupported_text_meaning.status is VideoQuestionStatus.UNSUPPORTED
    print("FULL, VISUAL, OBJECT, TEXT, TIME, AND UNSUPPORTED QUESTION TYPES: PASS")


def test_friday_routes_video_context_without_qwen_classification() -> None:
    knowledge = make_complete_result().knowledge
    engine, qwen, _ = engine_for(knowledge)
    reply = engine.process("What happened in this video?")
    assert reply == "The grounded video context is available."
    assert qwen.decisions == 0
    assert qwen.responses[-1][1].capability == VIDEO_QUESTION_CAPABILITY
    context = qwen.responses[-1][2].data["context"]
    assert context["visual_events"][0]["timestamp"] == 0.0
    assert context["visual_events"][0]["frame_index"] == 0
    assert context["visual_events"][0]["track_id"] is not None
    assert context["visual_events"][0]["label"]
    assert context["visual_events"][0]["confidence"] is not None
    assert context["visual_events"][0]["current_position"] is not None
    assert context["evidence_references"][0]["source_frame_index"] == 0

    text_reply = engine.process("What text was shown?")
    assert "EXIT" in text_reply
    text_context = qwen.responses[-1][2].data["context"]
    exit_observation = next(
        item for item in text_context["text_observations"] if item["text"] == "EXIT"
    )
    assert exit_observation["timestamp"] == 0.0
    assert exit_observation["frame_index"] == 0
    assert exit_observation["bbox"] is not None
    print("FRIDAY USES GROUNDED STRUCTURED CONTEXT, NOT RAW FRAMES: PASS")


def test_follow_up_referent_and_duplicate_labels() -> None:
    knowledge = make_complete_result().knowledge
    engine, qwen, _ = engine_for(knowledge)
    engine.process("What happened in this video?")
    engine.process("What about the person?")
    object_context = qwen.responses[-1][2].data["context"]
    person = object_context["visual_objects"][0]
    assert person["label"] == "person"
    person_track = person["track_id"]

    engine.process("When did it appear?")
    follow_up_context = qwen.responses[-1][2].data["context"]
    assert follow_up_context["selection"]["track_id"] == person_track
    assert all(item["track_id"] == person_track for item in follow_up_context["visual_events"])
    assert engine.memory.current_referent == "person"

    result = VideoQuestionInterface().query("What happened to the bottle?", knowledge)
    assert result.status is VideoQuestionStatus.INSUFFICIENT
    bottles = result.context["visual_objects"]
    assert len(bottles) == 2
    assert len({item["track_id"] for item in bottles}) == 2

    engine.process("What about the bottle?")
    assert qwen.responses[-1][2].data["evidence_status"] == "insufficient_context"
    engine.process("When did it appear?")
    assert qwen.responses[-1][2].data["evidence_status"] == "insufficient_context"
    print("FOLLOW-UP REFERENT AND DUPLICATE LABEL TRACK IDENTITY: PASS")


def test_unsupported_and_insufficient_context_are_explicit() -> None:
    engine, qwen, _ = engine_for(None)
    no_video_reply = engine.process("What happened in this video?")
    assert "don't have enough" in no_video_reply.lower()
    assert qwen.responses[-1][2].data["evidence_status"] == "insufficient_context"

    source = make_complete_result()
    engine, qwen, _ = engine_for(
        VideoKnowledge.from_layers(
            facts=source.facts.__class__(events=(), objects=()),
            episodes=(),
            evidence_index=source.evidence.__class__(),
            text_history=source.text_history.__class__(),
        )
    )
    no_evidence_reply = engine.process("What happened in this video?")
    assert "don't have enough" in no_evidence_reply.lower()
    assert qwen.responses[-1][2].data["evidence_status"] == "insufficient_context"

    engine, qwen, _ = engine_for(source.knowledge)
    unsupported_reply = engine.process(
        "What is the emotional meaning of the scene in this video?"
    )
    assert "don't have enough" in unsupported_reply.lower()
    assert qwen.responses[-1][2].data["evidence_status"] == "unsupported"
    print("UNSUPPORTED AND INSUFFICIENT VIDEO EVIDENCE: PASS")


def test_internal_question_identifiers_never_reach_user_reply() -> None:
    engine, qwen, _ = engine_for(make_complete_result().knowledge)
    reply = engine.process("What happened in this video?")
    assert "video.question" not in reply
    assert "full_video" not in reply
    assert "track_id" not in reply
    assert "frame_index" not in reply
    assert all(
        request.capability == VIDEO_QUESTION_CAPABILITY
        for _, request, _ in qwen.responses
    )
    print("INTERNAL VIDEO QUERY IDENTIFIERS STAY OUT OF USER REPLIES: PASS")


def test_non_video_messages_keep_existing_qwen_decision_flow() -> None:
    engine, qwen, _ = engine_for(make_complete_result().knowledge)
    assert engine.process("Tell me a joke.") == "Ordinary conversation."
    assert qwen.decisions == 1
    assert not qwen.responses
    assert engine.process("What about your day?") == "Ordinary conversation."
    assert qwen.decisions == 2
    assert engine.process("What happened to my package?") == "Ordinary conversation."
    assert qwen.decisions == 3

    engine.process("What happened in this video?")
    assert engine.process("What about your day?") == "Ordinary conversation."
    assert qwen.decisions == 4
    print("NON-VIDEO CONVERSATION FLOW IS UNCHANGED: PASS")


def main() -> None:
    test_question_classification_and_context_selection()
    test_friday_routes_video_context_without_qwen_classification()
    test_follow_up_referent_and_duplicate_labels()
    test_unsupported_and_insufficient_context_are_explicit()
    test_internal_question_identifiers_never_reach_user_reply()
    test_non_video_messages_keep_existing_qwen_decision_flow()
    print("FRIDAY VIDEO QUESTION INTEGRATION: PASS")


if __name__ == "__main__":
    main()
