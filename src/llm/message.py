from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationMemory:
    max_recent_messages: int = 12
    recent_messages: list[Message] = field(default_factory=list)
    current_topic: str | None = None
    current_referent: str | None = None
    current_task: str | None = None
    current_target: str | None = None
    timestamp: float = field(default_factory=time.time)

    def add_message(self, message: Message) -> None:
        self.recent_messages.append(message)
        limit = max(0, self.max_recent_messages)
        self.recent_messages = self.recent_messages[-limit:] if limit else []
        self.timestamp = message.timestamp

    def context_value(self, name: str) -> Any:
        if name not in {"current_topic", "current_referent", "current_task", "current_target"}:
            return None
        return getattr(self, name)