from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


class CapabilityKind(str, Enum):
    INFORMATION = "information"
    ACTION = "action"


@dataclass(frozen=True)
class CapabilityRequest:
    capability: str
    kind: CapabilityKind
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capability.strip():
            raise ValueError("A capability identifier is required")


@dataclass(frozen=True)
class GroundingData:
    scene_information: Mapping[str, Any] | None = None
    target_information: Mapping[str, Any] | None = None
    relative_depth: Mapping[str, Any] | None = None
    safety_state: Mapping[str, Any] | None = None
    location_information: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CapabilityResult:
    available: bool
    data: Mapping[str, Any] = field(default_factory=dict)
    grounding: GroundingData | None = None
    detail: str | None = None

    @classmethod
    def unavailable(cls, detail: str | None = None) -> CapabilityResult:
        return cls(available=False, detail=detail)


class CapabilityProvider(Protocol):
    def provide(self, request: CapabilityRequest) -> CapabilityResult:
        """Return subsystem data without generating user-facing language."""


class CapabilityRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, CapabilityProvider] = {}

    def register(self, capability: str, provider: CapabilityProvider) -> None:
        if not capability.strip():
            raise ValueError("A capability identifier is required")
        self._providers[capability] = provider

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        provider = self._providers.get(request.capability)
        if provider is None:
            return CapabilityResult.unavailable("No provider is registered")
        return provider.provide(request)