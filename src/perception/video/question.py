from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Callable

from src.perception.video.context import VideoContextBuilder
from src.perception.video.knowledge import VideoKnowledge


class VideoQuestionKind(str, Enum):
    FULL_VIDEO = "full_video"
    VISUAL_EVENTS = "visual_events"
    TIME_RANGE = "time_range"
    OBJECT = "object"
    TEXT = "text"
    UNSUPPORTED = "unsupported"


class VideoQuestionStatus(str, Enum):
    GROUNDED = "grounded"
    INSUFFICIENT = "insufficient_context"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class VideoQuestionResult:
    status: VideoQuestionStatus
    kind: VideoQuestionKind
    context: dict[str, object] | None
    detail: str | None = None
    referent_track_id: int | None = None
    referent_label: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "kind": self.kind.value,
            "detail": self.detail,
            "context": self.context,
        }


_RANGE_RE = re.compile(
    r"\b(?:between|from)\s+(\d+(?:\.\d+)?)\s*(?:and|to)\s*"
    r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",
    re.IGNORECASE,
)
_POINT_RE = re.compile(
    r"\b(?:around|about|at|near)\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",
    re.IGNORECASE,
)
_VIDEO_CUE_RE = re.compile(r"\b(?:video|clip|footage|recording)\b", re.IGNORECASE)
_FOLLOWUP_RE = re.compile(
    r"\b(?:it|that|this|same one)\b",
    re.IGNORECASE,
)
_WHAT_ABOUT_RE = re.compile(r"\bwhat about\b", re.IGNORECASE)
_TEXT_QUERY_RE = re.compile(
    r"\bwhat\s+(?:text|signs?|words?)\b.*"
    r"\b(?:shown|displayed|written|visible|say|said|read)\b"
    r"|\bwhat\s+(?:does|did)\s+(?:the\s+)?(?:sign|text|writing)\s+(?:say|read)\b"
    r"|\bcan you read (?:the\s+)?(?:text|sign|writing)\b",
    re.IGNORECASE,
)
_TEXT_EVENT_RE = re.compile(
    r"\b(?:when|where)\b.*\b(?:shown|displayed|written|appear(?:ed)?|visible)\b",
    re.IGNORECASE,
)
_TEXT_ENTITY_RE = re.compile(r"\b(?:text|signs?|words?|writing)\b", re.IGNORECASE)
_VISUAL_EVENTS_RE = re.compile(
    r"\bwhat\s+(?:objects?|things?)\s+(?:appeared|showed up|were visible)\b"
    r"|\bwhat\s+happened\s+(?:visually|to the objects)\b",
    re.IGNORECASE,
)
_OBJECT_QUERY_RE = re.compile(
    r"\b(?:what about|what happened to|when did|when was|where was|where did|did)\b",
    re.IGNORECASE,
)
_FULL_VIDEO_RE = re.compile(
    r"\b(?:what happened|tell me about|describe|summari[sz]e)\b.*"
    r"\b(?:video|clip|footage|recording)\b"
    r"|\bwhat(?:'s| is) in (?:this )?(?:video|clip)\b",
    re.IGNORECASE,
)
class VideoQuestionInterface:
    """Classify supported video questions and select grounded structured context."""

    def __init__(
        self,
        *,
        time_tolerance_seconds: float = 1.0,
        context_builder_factory: Callable[[VideoKnowledge], VideoContextBuilder] = VideoContextBuilder,
    ) -> None:
        if not math.isfinite(time_tolerance_seconds) or time_tolerance_seconds < 0:
            raise ValueError("time_tolerance_seconds must be finite and non-negative")
        self.time_tolerance_seconds = time_tolerance_seconds
        self.context_builder_factory = context_builder_factory

    @staticmethod
    def is_video_question(question: str, *, active_video_context: bool = False) -> bool:
        return VideoQuestionInterface.is_supported_video_question(
            question,
            active_video_context=active_video_context,
        )

    @staticmethod
    def is_supported_video_question(
        question: str,
        *,
        active_video_context: bool = False,
        known_referents: tuple[str, ...] = (),
    ) -> bool:
        if (
            _VIDEO_CUE_RE.search(question)
            or _RANGE_RE.search(question)
            or _POINT_RE.search(question)
            or _TEXT_QUERY_RE.search(question)
            or _VISUAL_EVENTS_RE.search(question)
        ):
            return True
        mentioned_referent = any(
            re.search(
                rf"(?<!\w){re.escape(label.casefold())}(?!\w)",
                question,
                flags=re.IGNORECASE,
            )
            for label in known_referents
            if label.strip()
        )
        if mentioned_referent and (
            _OBJECT_QUERY_RE.search(question) or _TEXT_EVENT_RE.search(question)
        ):
            return True
        if active_video_context and _FOLLOWUP_RE.search(question):
            return True
        return active_video_context and mentioned_referent and bool(
            _WHAT_ABOUT_RE.search(question)
        )

    def query(
        self,
        question: str,
        knowledge: VideoKnowledge,
        *,
        referent_track_id: int | None = None,
    ) -> VideoQuestionResult:
        normalized = " ".join(question.casefold().split())
        builder = self.context_builder_factory(knowledge)

        time_range = self._time_range(normalized)
        if time_range is not None:
            context = builder.for_time_range(*time_range)
            return self._result(
                VideoQuestionKind.TIME_RANGE,
                context,
                referent_track_id=referent_track_id,
                referent_label=self._label_for_track(knowledge, referent_track_id),
            )

        if _FULL_VIDEO_RE.search(normalized):
            return self._result(VideoQuestionKind.FULL_VIDEO, builder.build())

        mentioned_text = self._mentioned_text(normalized, knowledge)
        if (
            _TEXT_QUERY_RE.search(normalized)
            or (
                _TEXT_EVENT_RE.search(normalized)
                and (_TEXT_ENTITY_RE.search(normalized) or mentioned_text is not None)
            )
            or (mentioned_text is not None and _WHAT_ABOUT_RE.search(normalized))
        ):
            text = mentioned_text
            context = builder.for_text(text) if text else builder.for_all_text()
            return self._result(
                VideoQuestionKind.TEXT,
                context,
                referent_label=text,
            )

        if _VISUAL_EVENTS_RE.search(normalized):
            return self._result(VideoQuestionKind.VISUAL_EVENTS, builder.for_visual_events())

        labels = self._mentioned_labels(normalized, knowledge)
        if labels and _OBJECT_QUERY_RE.search(normalized):
            if len(labels) > 1:
                return VideoQuestionResult(
                    status=VideoQuestionStatus.INSUFFICIENT,
                    kind=VideoQuestionKind.OBJECT,
                    context=builder.for_visual_events(),
                    detail="The question mentions multiple object labels without a supported comparison query.",
                )
            selected_label = labels[0]
            matching = [
                item
                for item in knowledge.facts.objects
                if item.label.casefold() == selected_label.casefold()
            ]
            context = builder.for_label(selected_label)
            if len(matching) == 1:
                return self._result(
                    VideoQuestionKind.OBJECT,
                    context,
                    referent_track_id=matching[0].track_id,
                    referent_label=matching[0].label,
                )
            return VideoQuestionResult(
                status=VideoQuestionStatus.INSUFFICIENT,
                kind=VideoQuestionKind.OBJECT,
                context=context,
                detail="More than one track has that label; the question does not identify a unique object.",
                referent_label=selected_label,
            )

        if referent_track_id is not None and _FOLLOWUP_RE.search(normalized):
            context = builder.for_object(referent_track_id)
            return self._result(
                VideoQuestionKind.OBJECT,
                context,
                referent_track_id=referent_track_id,
                referent_label=self._label_for_track(knowledge, referent_track_id),
            )

        return VideoQuestionResult(
            status=VideoQuestionStatus.UNSUPPORTED,
            kind=VideoQuestionKind.UNSUPPORTED,
            context=None,
            detail="The question does not match a supported grounded video query.",
        )

    def _time_range(self, question: str) -> tuple[float, float] | None:
        match = _RANGE_RE.search(question)
        if match:
            start, end = float(match.group(1)), float(match.group(2))
            if start <= end:
                return start, end
            return end, start
        match = _POINT_RE.search(question)
        if match:
            point = float(match.group(1))
            return (
                max(0.0, point - self.time_tolerance_seconds),
                point + self.time_tolerance_seconds,
            )
        return None

    @staticmethod
    def _mentioned_labels(question: str, knowledge: VideoKnowledge) -> list[str]:
        labels = {item.label for item in knowledge.facts.objects if item.label.strip()}
        matches = [
            label
            for label in labels
            if re.search(
                rf"(?<!\w){re.escape(label.casefold())}(?!\w)",
                question,
                flags=re.IGNORECASE,
            )
        ]
        return sorted(matches, key=lambda item: (-len(item), item.casefold()))

    @staticmethod
    def _mentioned_text(question: str, knowledge: VideoKnowledge) -> str | None:
        texts = {
            item.text
            for item in knowledge.text_observations
            if item.text.strip()
        }
        matches = [
            text
            for text in texts
            if re.search(
                rf"(?<!\w){re.escape(text.casefold())}(?!\w)",
                question,
                flags=re.IGNORECASE,
            )
        ]
        return sorted(matches, key=lambda item: (-len(item), item.casefold()))[0] if matches else None

    @staticmethod
    def _label_for_track(knowledge: VideoKnowledge, track_id: int | None) -> str | None:
        if track_id is None:
            return None
        item = next(
            (item for item in knowledge.facts.objects if item.track_id == track_id),
            None,
        )
        return item.label if item is not None else None

    @staticmethod
    def _result(
        kind: VideoQuestionKind,
        context: dict[str, object],
        *,
        referent_track_id: int | None = None,
        referent_label: str | None = None,
    ) -> VideoQuestionResult:
        has_evidence = any(
            context.get(key)
            for key in (
                "visual_objects",
                "visual_events",
                "episodes",
                "text_observations",
                "evidence_references",
            )
        )
        text_history = context.get("text_history")
        if isinstance(text_history, dict):
            has_evidence = has_evidence or any(
                text_history.get(key) for key in ("episodes", "events")
            )
        return VideoQuestionResult(
            status=(
                VideoQuestionStatus.GROUNDED
                if has_evidence
                else VideoQuestionStatus.INSUFFICIENT
            ),
            kind=kind,
            context=context,
            detail=None if has_evidence else "No matching grounded video evidence is available.",
            referent_track_id=referent_track_id,
            referent_label=referent_label,
        )
