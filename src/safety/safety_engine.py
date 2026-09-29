from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SafetyState(str, Enum):
    CLEAR = "CLEAR"
    CAUTION = "CAUTION"
    STOP = "STOP"
    UNKNOWN = "UNKNOWN"


class AlertPriority(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    HIGH = "HIGH"


@dataclass(frozen=True)
class SafetyInput:
    obstacle_present: bool | None
    relative_depth: str | None
    confidence: float | None
    target_is_tracked: bool | None
    timestamp: float


@dataclass(frozen=True)
class SafetyResult:
    state: SafetyState
    priority: AlertPriority
    reason: str
    event_emitted: bool
    timestamp: float


class SafetyEngine:
    """Prototype event logic over abstract perception inputs; no model dependencies."""

    _STATE_LEVEL = {
        SafetyState.CLEAR: 0,
        SafetyState.CAUTION: 1,
        SafetyState.STOP: 2,
    }
    _STATE_AT_LEVEL = {level: state for state, level in _STATE_LEVEL.items()}

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        cooldown_seconds: float = 2.0,
        initial_state: SafetyState = SafetyState.UNKNOWN,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.state = initial_state
        self._last_event_signature: tuple[SafetyState, AlertPriority, str] | None = None
        self._last_event_timestamp: float | None = None

    @staticmethod
    def _priority_for(state: SafetyState) -> AlertPriority:
        if state is SafetyState.STOP:
            return AlertPriority.HIGH
        if state in {SafetyState.CAUTION, SafetyState.UNKNOWN}:
            return AlertPriority.CAUTION
        return AlertPriority.NORMAL

    def _assess(self, observation: SafetyInput) -> tuple[SafetyState, str]:
        if observation.obstacle_present is False:
            if observation.target_is_tracked is False:
                return SafetyState.CLEAR, "Target is not tracked; no obstacle evidence"
            return SafetyState.CLEAR, "No obstacle detected"
        if observation.obstacle_present is None:
            return SafetyState.UNKNOWN, "Obstacle presence is unavailable"
        if observation.confidence is None or observation.confidence < self.confidence_threshold:
            return SafetyState.CAUTION, "Obstacle detected with insufficient confidence"

        depth = (observation.relative_depth or "unknown").strip().lower()
        if depth == "relatively near":
            return SafetyState.STOP, "Near obstacle detected"
        if depth == "relatively far":
            return SafetyState.CAUTION, "Obstacle detected relatively far"
        if depth == "relatively middle-distance":
            return SafetyState.CAUTION, "Obstacle detected at relative middle-distance"
        return SafetyState.UNKNOWN, "Obstacle detected; relative depth is unavailable"

    def _stage_transition(self, desired: SafetyState) -> tuple[SafetyState, bool]:
        current = self.state
        if desired is SafetyState.UNKNOWN:
            return SafetyState.UNKNOWN, False
        if current is SafetyState.UNKNOWN:
            if desired is SafetyState.STOP:
                return SafetyState.CAUTION, True
            return desired, False

        current_level = self._STATE_LEVEL[current]
        desired_level = self._STATE_LEVEL[desired]
        if desired_level > current_level + 1:
            return self._STATE_AT_LEVEL[current_level + 1], True
        if desired_level < current_level - 1:
            return self._STATE_AT_LEVEL[current_level - 1], True
        return desired, False

    def evaluate(self, observation: SafetyInput) -> SafetyResult:
        desired_state, assessment_reason = self._assess(observation)
        next_state, staged = self._stage_transition(desired_state)
        previous_state = self.state
        self.state = next_state
        priority = self._priority_for(next_state)
        reason = assessment_reason
        if staged:
            reason = f"{assessment_reason}; state transition staged through {next_state.value}"

        signature = (next_state, priority, reason)
        state_changed = next_state is not previous_state
        outside_cooldown = (
            self._last_event_timestamp is None
            or observation.timestamp - self._last_event_timestamp >= self.cooldown_seconds
        )
        event_emitted = state_changed or outside_cooldown
        if event_emitted:
            self._last_event_signature = signature
            self._last_event_timestamp = observation.timestamp
        elif signature != self._last_event_signature and priority is AlertPriority.HIGH:
            event_emitted = True
            self._last_event_signature = signature
            self._last_event_timestamp = observation.timestamp

        return SafetyResult(
            state=next_state,
            priority=priority,
            reason=reason,
            event_emitted=event_emitted,
            timestamp=observation.timestamp,
        )
