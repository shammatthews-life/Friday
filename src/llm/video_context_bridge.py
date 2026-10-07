from __future__ import annotations

import re

from src.llm.message import ConversationMemory
from src.llm.tool_interface import CapabilityResult
from src.perception.video.knowledge import VideoKnowledge
from src.perception.video.question import (
    VideoQuestionInterface,
    VideoQuestionKind,
    VideoQuestionResult,
    VideoQuestionStatus,
)


VIDEO_QUESTION_CAPABILITY = "video.question"
_VIDEO_TASK = VIDEO_QUESTION_CAPABILITY
_FOLLOWUP_ONLY_RE = re.compile(
    r"\b(?:it|that|this|same one)\b|\bwhat about\b|\bwhen did it\b",
    re.IGNORECASE,
)


class VideoConversationBridge:
    """Route deterministic video context into FRIDAY's existing LLM response path."""

    def __init__(
        self,
        knowledge: VideoKnowledge | None = None,
        *,
        question_interface: VideoQuestionInterface | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.question_interface = question_interface or VideoQuestionInterface()
        self._referent_track_id: int | None = None
        self._ambiguous_label: str | None = None

    def update_knowledge(self, knowledge: VideoKnowledge | None) -> None:
        self.knowledge = knowledge
        self._referent_track_id = None
        self._ambiguous_label = None

    def handle_question(
        self,
        question: str,
        memory: ConversationMemory,
    ) -> CapabilityResult | None:
        active_context = memory.current_task == _VIDEO_TASK
        known_referents = ()
        if self.knowledge is not None:
            known_referents = tuple(
                item.label for item in self.knowledge.facts.objects
            ) + tuple(
                item.text for item in self.knowledge.text_observations
            )
        if not self.question_interface.is_supported_video_question(
            question,
            active_video_context=active_context,
            known_referents=known_referents,
        ):
            return None

        if self.knowledge is None:
            result = VideoQuestionResult(
                status=VideoQuestionStatus.INSUFFICIENT,
                kind=VideoQuestionKind.UNSUPPORTED,
                context=None,
                detail="No video analysis is available for this question.",
            )
        elif self._ambiguous_label and _FOLLOWUP_ONLY_RE.search(question):
            result = VideoQuestionResult(
                status=VideoQuestionStatus.INSUFFICIENT,
                kind=VideoQuestionKind.OBJECT,
                context=self.question_interface.context_builder_factory(
                    self.knowledge
                ).for_label(self._ambiguous_label),
                detail="The prior object reference matches multiple tracks.",
                referent_label=self._ambiguous_label,
            )
        else:
            result = self.question_interface.query(
                question,
                self.knowledge,
                referent_track_id=self._referent_track_id,
            )

        if result.kind is VideoQuestionKind.OBJECT:
            self._referent_track_id = result.referent_track_id
            self._ambiguous_label = (
                result.referent_label
                if result.status is VideoQuestionStatus.INSUFFICIENT
                and result.referent_track_id is None
                else None
            )
            memory.current_referent = (
                result.referent_label
                if result.referent_track_id is not None
                else None
            )
        elif result.kind in {
            VideoQuestionKind.FULL_VIDEO,
            VideoQuestionKind.TEXT,
            VideoQuestionKind.VISUAL_EVENTS,
            VideoQuestionKind.TIME_RANGE,
        }:
            self._referent_track_id = result.referent_track_id
            self._ambiguous_label = None
            if result.referent_label:
                memory.current_referent = result.referent_label
        memory.current_topic = "video"
        memory.current_task = _VIDEO_TASK
        return CapabilityResult(
            available=True,
            data={
                "evidence_status": result.status.value,
                "context": result.context,
                "detail": result.detail,
            },
            detail=result.detail,
        )
