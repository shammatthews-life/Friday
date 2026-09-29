from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.safety.safety_engine import AlertPriority, SafetyEngine, SafetyInput, SafetyState


def observation(
    timestamp: float,
    obstacle_present: bool | None,
    relative_depth: str | None,
    confidence: float | None,
    target_is_tracked: bool | None = True,
) -> SafetyInput:
    return SafetyInput(
        obstacle_present=obstacle_present,
        relative_depth=relative_depth,
        confidence=confidence,
        target_is_tracked=target_is_tracked,
        timestamp=timestamp,
    )


def report(name: str, sensor_input: SafetyInput, result) -> None:
    print(
        f"{name}: input={sensor_input} | state={result.state.value} "
        f"| priority={result.priority.value} | reason={result.reason} "
        f"| event={'emitted' if result.event_emitted else 'not emitted'}"
    )


def main() -> None:
    engine = SafetyEngine(cooldown_seconds=2.0)

    input_a = observation(0.0, False, "unknown", None)
    result_a = engine.evaluate(input_a)
    report("A NO OBSTACLE", input_a, result_a)
    assert (result_a.state, result_a.priority) == (SafetyState.CLEAR, AlertPriority.NORMAL)

    input_b = observation(1.0, True, "relatively far", 0.92)
    result_b = engine.evaluate(input_b)
    report("B FAR OBSTACLE", input_b, result_b)
    assert (result_b.state, result_b.priority) == (SafetyState.CAUTION, AlertPriority.CAUTION)

    input_c = observation(2.0, True, "relatively near", 0.94)
    result_c = engine.evaluate(input_c)
    report("C NEAR OBSTACLE", input_c, result_c)
    assert (result_c.state, result_c.priority) == (SafetyState.STOP, AlertPriority.HIGH)

    repeated_results = []
    for timestamp in (2.5, 3.0, 3.5):
        repeated_input = observation(timestamp, True, "relatively near", 0.94)
        repeated_result = engine.evaluate(repeated_input)
        repeated_results.append(repeated_result)
        report(f"E REPEATED STOP t={timestamp}", repeated_input, repeated_result)
    assert not any(result.event_emitted for result in repeated_results)

    cooldown_expired_input = observation(4.1, True, "relatively near", 0.94)
    cooldown_expired_result = engine.evaluate(cooldown_expired_input)
    report("E COOLDOWN EXPIRED", cooldown_expired_input, cooldown_expired_result)
    assert cooldown_expired_result.state is SafetyState.STOP
    assert cooldown_expired_result.event_emitted

    input_f = observation(4.2, True, "relatively far", 0.92)
    result_f = engine.evaluate(input_f)
    report("F STOP TO CAUTION", input_f, result_f)
    assert (result_f.state, result_f.priority) == (SafetyState.CAUTION, AlertPriority.CAUTION)
    assert result_f.event_emitted

    input_g = observation(4.3, False, "unknown", None)
    result_g = engine.evaluate(input_g)
    report("G CAUTION TO CLEAR", input_g, result_g)
    assert (result_g.state, result_g.priority) == (SafetyState.CLEAR, AlertPriority.NORMAL)
    assert result_g.event_emitted

    staged_engine = SafetyEngine(initial_state=SafetyState.CLEAR)
    first_near_input = observation(7.0, True, "relatively near", 0.94)
    first_near_result = staged_engine.evaluate(first_near_input)
    report("ESCALATION STEP 1", first_near_input, first_near_result)
    assert first_near_result.state is SafetyState.CAUTION
    second_near_input = observation(7.1, True, "relatively near", 0.94)
    second_near_result = staged_engine.evaluate(second_near_input)
    report("ESCALATION STEP 2", second_near_input, second_near_result)
    assert second_near_result.state is SafetyState.STOP

    unknown_engine = SafetyEngine(initial_state=SafetyState.CLEAR)
    input_d = observation(4.0, True, "unknown", 0.90)
    result_d = unknown_engine.evaluate(input_d)
    report("D UNKNOWN DEPTH", input_d, result_d)
    assert (result_d.state, result_d.priority) == (SafetyState.UNKNOWN, AlertPriority.CAUTION)

    low_confidence_engine = SafetyEngine(initial_state=SafetyState.CLEAR)
    input_h = observation(5.0, True, "relatively near", 0.30)
    result_h = low_confidence_engine.evaluate(input_h)
    report("H LOW CONFIDENCE", input_h, result_h)
    assert (result_h.state, result_h.priority) == (SafetyState.CAUTION, AlertPriority.CAUTION)

    lost_target_engine = SafetyEngine(initial_state=SafetyState.CLEAR)
    input_i = observation(6.0, False, "unknown", None, target_is_tracked=False)
    result_i = lost_target_engine.evaluate(input_i)
    report("I TARGET LOST WITHOUT OBSTACLE", input_i, result_i)
    assert result_i.state is not SafetyState.STOP
    assert result_i.priority is not AlertPriority.HIGH

    print("UNKNOWN-DEPTH POLICY: UNKNOWN / CAUTION when obstacle presence is confirmed but relative depth is unavailable.")
    print("SAFETY ENGINE LOGIC: PASS")


if __name__ == "__main__":
    main()
