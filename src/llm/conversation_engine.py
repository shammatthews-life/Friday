from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Mapping, Protocol

from src.llm.message import ConversationMemory, Message, MessageRole
from src.llm.tool_interface import (
    CapabilityRegistry,
    CapabilityKind,
    CapabilityRequest,
    CapabilityResult,
)


class DecisionKind(str, Enum):
    ANSWER = "answer"
    INFORMATION = "information"
    ACTION = "action"


@dataclass(frozen=True)
class LLMDecision:
    kind: DecisionKind
    response: str = ""
    capability_request: CapabilityRequest | None = None
    topic: str | None = None
    current_referent: str | None = None
    clarification: str | None = None

    def __post_init__(self) -> None:
        if self.kind is DecisionKind.ANSWER and self.capability_request is not None:
            raise ValueError("An answer decision cannot include a capability request")
        if self.kind is not DecisionKind.ANSWER and self.capability_request is None:
            raise ValueError("Information and action decisions require a capability request")
        if self.capability_request is not None:
            expected_kind = (
                CapabilityKind.INFORMATION
                if self.kind is DecisionKind.INFORMATION
                else CapabilityKind.ACTION
            )
            if self.capability_request.kind is not expected_kind:
                raise ValueError("Decision kind must match the capability request kind")


class LLM(Protocol):
    def decide(self, user_message: str, memory: ConversationMemory) -> LLMDecision:
        """Choose a normal answer or a structured capability request."""

    def respond_to_capability(
        self,
        user_message: str,
        request: CapabilityRequest,
        result: CapabilityResult,
        memory: ConversationMemory,
    ) -> str:
        """Compose a natural response grounded in the returned subsystem data."""


class ConversationEngine:
    def __init__(
        self,
        llm: LLM,
        capabilities: CapabilityRegistry | None = None,
        memory: ConversationMemory | None = None,
        clarification_response: str = "Which item would you like me to guide you to?",
    ) -> None:
        self.llm = llm
        self.capabilities = capabilities or CapabilityRegistry()
        self.memory = memory or ConversationMemory()
        self.clarification_response = clarification_response

    def process(self, user_message: str) -> str:
        text = user_message.strip()
        if not text:
            return "What would you like to talk about?"

        self.memory.add_message(Message(MessageRole.USER, text))
        decision = self.llm.decide(text, self.memory)
        if decision.topic:
            self.memory.current_topic = decision.topic
        if decision.current_referent:
            self.memory.current_referent = decision.current_referent

        request = decision.capability_request
        if decision.kind is DecisionKind.ANSWER or request is None:
            response = decision.response
        else:
            resolved_arguments, missing_reference = self._resolve_references(
                request.arguments
            )
            if missing_reference:
                response = decision.clarification or self.clarification_response
            else:
                resolved_request = CapabilityRequest(
                    capability=request.capability,
                    kind=request.kind,
                    arguments=resolved_arguments,
                )
                self._record_task_and_target(resolved_request)
                result = self.capabilities.invoke(resolved_request)
                response = self.llm.respond_to_capability(
                    text, resolved_request, result, self.memory
                )

        if not response.strip():
            response = "I'm here. What would you like to talk about?"
        self.memory.add_message(
            Message(MessageRole.ASSISTANT, response, timestamp=time.time())
        )
        return response

    def _resolve_references(self, value: Any) -> tuple[Any, bool]:
        if isinstance(value, Mapping):
            if set(value) == {"$context"}:
                resolved = self.memory.context_value(str(value["$context"]))
                return resolved, resolved is None
            resolved_mapping: dict[str, Any] = {}
            for key, item in value.items():
                resolved_item, missing = self._resolve_references(item)
                if missing:
                    return None, True
                resolved_mapping[str(key)] = resolved_item
            return resolved_mapping, False
        if isinstance(value, list):
            resolved_items = []
            for item in value:
                resolved_item, missing = self._resolve_references(item)
                if missing:
                    return None, True
                resolved_items.append(resolved_item)
            return resolved_items, False
        return value, False

    def _record_task_and_target(self, request: CapabilityRequest) -> None:
        self.memory.current_task = request.capability
        for key in ("target", "object", "label"):
            value = request.arguments.get(key)
            if isinstance(value, str) and value:
                self.memory.current_target = value
                self.memory.current_referent = value
                break