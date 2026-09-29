from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.guidance import GuidanceEngine, GuidanceState
from src.search.object_search import TargetState


def check_case(name: str, result, expected_states: tuple[GuidanceState, ...], expected_text: str) -> None:
    assert result.states == expected_states, (name, result.states)
    assert result.text == expected_text, (name, result.text)
    print(f"{name}: {result.target_state.value} | {','.join(state.value for state in result.states)} | {result.text}")


def main() -> None:
    engine = GuidanceEngine(initial_target_state=TargetState.LOCKED)
    check_case(
        "A LEFT + FAR",
        engine.update("left", "relatively far"),
        (GuidanceState.TARGET_LEFT, GuidanceState.TARGET_FAR),
        "Move slightly left. The target appears relatively far.",
    )

    engine = GuidanceEngine(initial_target_state=TargetState.LOCKED)
    check_case(
        "B RIGHT + MIDDLE",
        engine.update("right", "relatively middle-distance"),
        (GuidanceState.TARGET_RIGHT,),
        "Move slightly right. The target appears relatively middle-distance.",
    )

    engine = GuidanceEngine(initial_target_state=TargetState.LOCKED)
    check_case(
        "C CENTER + NEAR",
        engine.update("center", "relatively near"),
        (GuidanceState.TARGET_CENTER, GuidanceState.TARGET_NEAR),
        "The target is centered. The target appears relatively near.",
    )

    engine = GuidanceEngine(initial_target_state=TargetState.LOCKED)
    check_case(
        "D CENTER + UNKNOWN",
        engine.update("center", "unknown"),
        (GuidanceState.TARGET_CENTER,),
        "The target is centered, but relative depth is unavailable.",
    )

    engine = GuidanceEngine(initial_target_state=TargetState.LOCKED)
    check_case(
        "E LOST",
        engine.update(target_state=TargetState.LOST),
        (GuidanceState.TARGET_LOST,),
        "I lost track of the target.",
    )

    searching = GuidanceEngine()
    check_case(
        "F SEARCHING",
        searching.update(target_state=TargetState.SEARCHING),
        (GuidanceState.TARGET_UNKNOWN,),
        "I am searching for the target.",
    )

    transitions = GuidanceEngine()
    found = transitions.update(target_state=TargetState.FOUND)
    assert found.target_state is TargetState.FOUND
    locked = transitions.update("center", "unknown", TargetState.LOCKED)
    assert locked.target_state is TargetState.LOCKED
    print("LIFECYCLE SEARCHING -> FOUND -> LOCKED: PASS")
    print("GUIDANCE LOGIC: PASS")


if __name__ == "__main__":
    main()
