from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.search.object_search import TargetState


class GuidanceState(str, Enum):
    TARGET_LEFT = "TARGET_LEFT"
    TARGET_CENTER = "TARGET_CENTER"
    TARGET_RIGHT = "TARGET_RIGHT"
    TARGET_NEAR = "TARGET_NEAR"
    TARGET_FAR = "TARGET_FAR"
    TARGET_LOST = "TARGET_LOST"
    TARGET_UNKNOWN = "TARGET_UNKNOWN"


@dataclass(frozen=True)
class GuidanceResult:
    target_state: TargetState
    states: tuple[GuidanceState, ...]
    text: str

    @property
    def guidance_state(self) -> GuidanceState:
        return self.states[0]


class GuidanceEngine:
    """Deterministic target alignment cues; not navigation or safety control."""

    _TRANSITIONS = {
        TargetState.SEARCHING: {TargetState.SEARCHING, TargetState.FOUND, TargetState.LOST},
        TargetState.FOUND: {TargetState.FOUND, TargetState.LOCKED, TargetState.LOST},
        TargetState.LOCKED: {TargetState.LOCKED, TargetState.LOST},
        TargetState.LOST: {TargetState.LOST, TargetState.SEARCHING},
    }

    def __init__(self, initial_target_state: TargetState = TargetState.SEARCHING) -> None:
        self.target_state = self._coerce_target_state(initial_target_state)

    @staticmethod
    def _coerce_target_state(value: TargetState | str) -> TargetState:
        if isinstance(value, TargetState):
            return value
        return TargetState(str(value).upper())

    def _transition(self, requested: TargetState) -> None:
        if requested not in self._TRANSITIONS[self.target_state]:
            raise ValueError(
                f"Invalid target-state transition: {self.target_state.value} -> {requested.value}"
            )
        self.target_state = requested

    def update(
        self,
        target_position: str | None = None,
        relative_depth: str = "unknown",
        target_state: TargetState | str | None = None,
    ) -> GuidanceResult:
        if target_state is not None:
            self._transition(self._coerce_target_state(target_state))

        if self.target_state is TargetState.LOST:
            return GuidanceResult(
                self.target_state, (GuidanceState.TARGET_LOST,), "I lost track of the target."
            )
        if self.target_state is TargetState.SEARCHING:
            return GuidanceResult(
                self.target_state, (GuidanceState.TARGET_UNKNOWN,), "I am searching for the target."
            )
        if self.target_state is TargetState.FOUND:
            return GuidanceResult(
                self.target_state,
                (GuidanceState.TARGET_UNKNOWN,),
                "I found the target; alignment guidance is not locked yet.",
            )

        position = (target_position or "unknown").strip().lower()
        depth = (relative_depth or "unknown").strip().lower()
        position_states = {
            "left": (GuidanceState.TARGET_LEFT, "Move slightly left."),
            "center": (GuidanceState.TARGET_CENTER, "The target is centered."),
            "right": (GuidanceState.TARGET_RIGHT, "Move slightly right."),
        }
        if position not in position_states:
            return GuidanceResult(
                self.target_state,
                (GuidanceState.TARGET_UNKNOWN,),
                "I cannot determine the target position.",
            )

        position_state, alignment_text = position_states[position]
        states = [position_state]
        depth_text = ""
        if depth == "relatively near":
            states.append(GuidanceState.TARGET_NEAR)
            depth_text = " The target appears relatively near."
        elif depth == "relatively far":
            states.append(GuidanceState.TARGET_FAR)
            depth_text = " The target appears relatively far."
        elif depth == "relatively middle-distance":
            depth_text = " The target appears relatively middle-distance."
        elif depth != "unknown":
            depth = "unknown"

        if position == "center" and depth == "unknown":
            text = "The target is centered, but relative depth is unavailable."
        else:
            text = alignment_text + depth_text
        return GuidanceResult(self.target_state, tuple(states), text)
