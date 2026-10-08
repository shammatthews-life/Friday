from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from test_qwen_runtime_profile import (  # noqa: E402
    _run_configuration,
    _selected_device_is_rtx,
    _visual_mean,
)


REPORT_PATH = ROOT / "data" / "evaluation" / "qwen_gpu_layer_sweep.md"
SMOKE_REPORT_PATH = ROOT / "data" / "evaluation" / "qwen_video_smoke_test.md"
QUESTIONS = (
    "What happened in this video?",
    "What about the sports ball?",
    "When did it appear?",
    "What text was shown?",
)
BASE_CONFIGURATIONS = (28, 30, 32)
RUNS_PER_CONFIGURATION = 2
RTX_VRAM_TOTAL_MIB = 6141
MIN_PROJECTED_HEADROOM_MIB = 1024
NEXT_LAYER_STEP = 4
METRICS = (
    ("startup_seconds", "Server/model startup (s)"),
    ("prompt_tokens", "Prompt tokens"),
    ("prompt_processing_estimate_seconds", "Prompt/prefill (s)"),
    ("first_token_seconds", "First token (s)"),
    ("completion_seconds", "Completion (s)"),
    ("total_response_seconds", "Total response (s)"),
    ("completion_tokens", "Generated tokens"),
    ("tokens_per_second", "Generation tok/s"),
)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _configuration_metrics(reports: list[dict[str, Any]]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for key, _ in METRICS:
        if key == "startup_seconds":
            values = [
                float(report["runtime"]["server_model_load_seconds"])
                for report in reports
                if report.get("runtime", {}).get("server_model_load_seconds")
                is not None
            ]
            metrics[key] = _mean(values)
        else:
            metrics[key] = _visual_mean(reports, key)
    return metrics


def _run_stable(report: dict[str, Any]) -> bool:
    turns = report.get("turns", [])
    all_checks = len(turns) == len(QUESTIONS) and all(
        all(turn.get("checks", {}).values())
        and turn.get("first_token_seconds") is not None
        and turn.get("completion_seconds") is not None
        and turn.get("prompt_tokens") is not None
        and turn.get("completion_tokens") is not None
        and turn.get("tokens_per_second") is not None
        for turn in turns
    )
    return (
        report.get("status") == "PASS"
        and report.get("process_exit_code") == 0
        and report.get("video", {}).get("session_status") == "completed"
        and report.get("runtime", {}).get("base_url_loopback") is True
        and _selected_device_is_rtx(report)
        and bool(report.get("memory", {}).get("nvidia", {}).get("sample_count"))
        and all_checks
    )


def _max_gpu_memory(reports: list[dict[str, Any]]) -> float | None:
    values = [
        float(report["memory"]["nvidia"]["peak_memory_used_mib"])
        for report in reports
        if report.get("memory", {}).get("nvidia", {}).get("peak_memory_used_mib")
        is not None
    ]
    return max(values) if values else None


def _conditional_next_layer_allowed(
    runs: dict[int, list[dict[str, Any]]],
) -> tuple[bool, str]:
    thirty_two = runs.get(32, [])
    if len(thirty_two) != RUNS_PER_CONFIGURATION or not all(
        _run_stable(report) for report in thirty_two
    ):
        return False, "Skipped: both 32-layer runs must be stable and pass grounding."
    peak_32 = _max_gpu_memory(thirty_two)
    peak_28 = _max_gpu_memory(runs.get(28, []))
    if peak_32 is None or peak_28 is None:
        return False, "Skipped: RTX VRAM sampling is incomplete."
    observed_delta = max(0.0, peak_32 - peak_28)
    projected_peak = peak_32 + observed_delta
    projected_headroom = RTX_VRAM_TOTAL_MIB - projected_peak
    if projected_headroom < MIN_PROJECTED_HEADROOM_MIB:
        return (
            False,
            f"Skipped: conservative 36-layer projection leaves only "
            f"{projected_headroom:.0f} MiB VRAM headroom "
            f"(minimum {MIN_PROJECTED_HEADROOM_MIB} MiB).",
        )
    return (
        True,
        f"Eligible: projected 36-layer peak {projected_peak:.0f} MiB leaves "
        f"{projected_headroom:.0f} MiB headroom.",
    )


def _validate_report(report: dict[str, Any], layers: int, run_index: int) -> list[str]:
    failures: list[str] = []
    if not _run_stable(report):
        failures.append(
            f"{layers}-layer run {run_index} was unstable, failed grounding, "
            "or lacked required timing/GPU metrics."
        )
    if report.get("runtime", {}).get("gpu_layers") != layers:
        failures.append(
            f"{layers}-layer run {run_index} reported an unexpected GPU-layer count."
        )
    return failures


def _visual_variability(runs: dict[int, list[dict[str, Any]]], layers: int) -> str:
    reports = runs.get(layers, [])
    first_token = [
        float(turn["first_token_seconds"])
        for report in reports
        for turn in report.get("turns", [])
        if turn.get("question") in QUESTIONS[:3]
        and turn.get("first_token_seconds") is not None
    ]
    total = [
        float(turn["total_response_seconds"])
        for report in reports
        for turn in report.get("turns", [])
        if turn.get("question") in QUESTIONS[:3]
        and turn.get("total_response_seconds") is not None
    ]
    return (
        f"first-token range {min(first_token):.2f}–{max(first_token):.2f}s; "
        f"total-response range {min(total):.2f}–{max(total):.2f}s"
        if first_token and total
        else "metrics unavailable"
    )


def _render_report(
    runs: dict[int, list[dict[str, Any]]],
    failures: list[str],
    conditional_result: str,
    tested_next_layers: bool,
) -> str:
    tested_layers = sorted(runs)
    lines = [
        "# Qwen3-8B RTX 4050 Vulkan GPU-layer sweep",
        "",
        f"- Result: **{'PASS' if not failures else 'FAIL'}**",
        "- Model: Qwen3-8B Q4_K_M",
        "- Runtime: llama.cpp `0.5.0-dev` build 11146",
        "- Selected GPU: NVIDIA GeForce RTX 4050 Laptop GPU (`Vulkan1`); AMD Radeon 740M is `Vulkan0` and was not selected.",
        "- Fixed settings: context 4096; batch/ubatch 128/32; 8 CPU threads; flash attention auto; default KV cache.",
        f"- Repetitions per configuration: {RUNS_PER_CONFIGURATION}",
        f"- Tested GPU layers: {', '.join(map(str, tested_layers))}",
        "- Test order: 28, 30, 32, 32, 30, 28; optional 36-layer check only after the 32-layer stability/headroom gate.",
        f"- Conditional next-layer decision: {conditional_result}",
        "",
        "All values below are arithmetic means across the two full four-turn runs; only the three visual turns are averaged for the main latency comparison. Per-turn values are shown separately.",
        "",
        "## Configuration summary",
        "",
        "| GPU layers | Startup (s) | Prompt tokens | Prefill (s) | First token (s) | Completion (s) | Total response (s) | Generated tokens | tok/s | Peak process RSS (MiB) | Peak RTX VRAM (MiB) | Mean GPU util (%) | Stability |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    baseline = _configuration_metrics(runs.get(28, []))
    for layers in tested_layers:
        reports = runs[layers]
        metrics = _configuration_metrics(reports)
        rss = _mean(
            [
                float(report["memory"]["peak_process_tree_rss_mib"])
                for report in reports
                if report.get("memory", {}).get("peak_process_tree_rss_mib")
                is not None
            ]
        )
        gpu = [
            report.get("memory", {}).get("nvidia", {})
            for report in reports
        ]
        peak_vram = _max_gpu_memory(reports)
        mean_util = _mean(
            [
                float(item["average_utilization_percent"])
                for item in gpu
                if item.get("average_utilization_percent") is not None
            ]
        )
        stable = all(_run_stable(report) for report in reports) and len(
            reports
        ) == RUNS_PER_CONFIGURATION
        row = [
            layers,
            metrics["startup_seconds"],
            metrics["prompt_tokens"],
            metrics["prompt_processing_estimate_seconds"],
            metrics["first_token_seconds"],
            metrics["completion_seconds"],
            metrics["total_response_seconds"],
            metrics["completion_tokens"],
            metrics["tokens_per_second"],
            rss,
            peak_vram,
            mean_util,
        ]
        lines.append(
            "| "
            + " | ".join(
                f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"
                for value in row
            )
            + f" | {'PASS' if stable else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Change from 28 layers",
            "",
            "| GPU layers | First-token change | Total-response change | Prefill change | tok/s change | Peak RTX VRAM (MiB) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layers in tested_layers:
        metrics = _configuration_metrics(runs[layers])
        changes = []
        for key in (
            "first_token_seconds",
            "total_response_seconds",
            "prompt_processing_estimate_seconds",
        ):
            base = baseline.get(key)
            current = metrics.get(key)
            changes.append(
                f"{(current - base) / base * 100:+.1f}%"
                if base not in (None, 0) and current is not None
                else "n/a"
            )
        base_rate = baseline.get("tokens_per_second")
        current_rate = metrics.get("tokens_per_second")
        rate_change = (
            f"{(current_rate - base_rate) / base_rate * 100:+.1f}%"
            if base_rate not in (None, 0) and current_rate is not None
            else "n/a"
        )
        lines.append(
            f"| {layers} | {changes[0]} | {changes[1]} | {changes[2]} | "
            f"{rate_change} | {_max_gpu_memory(runs[layers]) or 'n/a'} |"
        )

    lines.extend(
        [
            "",
            "## Same-turn results",
            "",
            "| GPU layers | Question | Prompt tokens | Prefill (s) | First token (s) | Completion (s) | Total (s) | Generated tokens | tok/s |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layers in tested_layers:
        for question in QUESTIONS:
            turns = [
                turn
                for report in runs[layers]
                for turn in report.get("turns", [])
                if turn.get("question") == question
            ]
            values = []
            for key in (
                "prompt_tokens",
                "prompt_processing_estimate_seconds",
                "first_token_seconds",
                "completion_seconds",
                "total_response_seconds",
                "completion_tokens",
                "tokens_per_second",
            ):
                values.append(
                    _mean(
                        [
                            float(turn[key])
                            for turn in turns
                            if turn.get(key) is not None
                        ]
                    )
                )
            lines.append(
                f"| {layers} | {question} | "
                + " | ".join(
                    f"{value:.2f}" if value is not None else "n/a"
                    for value in values
                )
                + " |"
            )

    lines.extend(["", "## Per-run stability", ""])
    for layers in tested_layers:
        for index, report in enumerate(runs[layers], start=1):
            checks = all(
                all(turn.get("checks", {}).values())
                for turn in report.get("turns", [])
            )
            gpu = report.get("memory", {}).get("nvidia", {})
            lines.append(
                f"- {layers} layers, run {index}: "
                f"{'PASS' if _run_stable(report) else 'FAIL'}; grounding "
                f"{'PASS' if checks else 'FAIL'}; "
                f"peak RSS {report.get('memory', {}).get('peak_process_tree_rss_mib')} MiB; "
                f"peak RTX VRAM {gpu.get('peak_memory_used_mib')} / "
                f"{RTX_VRAM_TOTAL_MIB} MiB; "
                f"GPU samples {gpu.get('sample_count')}."
            )

    stable_layers = [
        layer
        for layer in tested_layers
        if len(runs[layer]) == RUNS_PER_CONFIGURATION
        and all(_run_stable(report) for report in runs[layer])
    ]
    safest = min(stable_layers) if stable_layers else None
    visual_first = {
        layer: _visual_mean(runs[layer], "first_token_seconds")
        for layer in stable_layers
    }
    visual_total = {
        layer: _visual_mean(runs[layer], "total_response_seconds")
        for layer in stable_layers
    }
    best = (
        min(stable_layers, key=lambda layer: (visual_first[layer], visual_total[layer]))
        if stable_layers
        else None
    )
    improvement_from_28 = (
        (visual_first[28] - visual_first[best]) / visual_first[28] * 100
        if best is not None
        and 28 in visual_first
        and visual_first[28] not in (None, 0)
        else None
    )
    max_vram_best = _max_gpu_memory(runs.get(best, [])) if best is not None else None
    headroom = (
        RTX_VRAM_TOTAL_MIB - max_vram_best if max_vram_best is not None else None
    )
    trend_lines = []
    for lower, upper in zip(stable_layers, stable_layers[1:]):
        lower_first = visual_first.get(lower)
        upper_first = visual_first.get(upper)
        lower_total = visual_total.get(lower)
        upper_total = visual_total.get(upper)
        if None in (lower_first, upper_first, lower_total, upper_total):
            continue
        first_delta = (upper_first - lower_first) / lower_first * 100
        total_delta = (upper_total - lower_total) / lower_total * 100
        if first_delta <= -5 and total_delta <= -5:
            description = "clear improvement"
        elif first_delta >= 5 or total_delta >= 5:
            description = "regression in at least one primary latency"
        else:
            description = "plateau / within 5% on primary latencies"
        trend_lines.append(
            f"- {lower} → {upper} layers: first token {first_delta:+.1f}%, "
            f"total response {total_delta:+.1f}% ({description})."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best measured setting: {best if best is not None else 'undetermined'} GPU layers, ranked by mean visual first-token latency then total response among stable configurations.",
            f"- Safest tested setting: {safest if safest is not None else 'undetermined'} GPU layers (lowest tested offload with all runs stable).",
            f"- Best vs. 28 layers: {improvement_from_28:.1f}% lower mean visual first-token latency."
            if improvement_from_28 is not None
            else "- Best vs. 28 layers: undetermined.",
            f"- Approximate RTX VRAM headroom at best setting: {headroom:.0f} MiB."
            if headroom is not None
            else "- Approximate RTX VRAM headroom at best setting: unavailable.",
            "- Peak process RSS and VRAM are sampled maxima, not reserved-memory guarantees.",
            *trend_lines,
            "- Plateau/regression thresholds: changes within ±5% on both primary latencies count as plateau; a ≥5% increase in either counts as regression.",
            "- Production/default GPU layers were not changed.",
            "",
            "## Failures",
            "",
        ]
    )
    lines.extend(f"- {failure}" for failure in failures) if failures else lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> int:
    runs: dict[int, list[dict[str, Any]]] = {layer: [] for layer in BASE_CONFIGURATIONS}
    failures: list[str] = []
    conditional_result = "Not evaluated."
    tested_next_layers = False
    original_smoke_report = (
        SMOKE_REPORT_PATH.read_bytes() if SMOKE_REPORT_PATH.is_file() else None
    )
    try:
        with tempfile.TemporaryDirectory(prefix="friday-qwen-gpu-sweep-") as temporary:
            temporary_path = Path(temporary)
            sequence = (28, 30, 32, 32, 30, 28)
            occurrences: dict[int, int] = {layer: 0 for layer in BASE_CONFIGURATIONS}
            for sequence_index, layers in enumerate(sequence, start=1):
                occurrences[layers] += 1
                run_index = occurrences[layers]
                json_path = temporary_path / f"run-{sequence_index}.json"
                label = f"GPU layer sweep: {layers}"
                print(
                    f"\n=== Sweep {sequence_index}/{len(sequence)}: "
                    f"{layers} GPU layers, run {run_index}/{RUNS_PER_CONFIGURATION} ==="
                )
                report = _run_configuration(
                    label,
                    layers,
                    run_index,
                    json_path,
                )
                runs[layers].append(report)
                failures.extend(_validate_report(report, layers, run_index))
                if not _run_stable(report):
                    conditional_result = (
                        "Skipped higher-layer test because a tested configuration "
                        "failed stability or grounding checks."
                    )
                    break

            if not failures:
                allowed, conditional_result = _conditional_next_layer_allowed(runs)
                if allowed:
                    tested_next_layers = True
                    runs[36] = []
                    for run_index in range(1, RUNS_PER_CONFIGURATION + 1):
                        json_path = temporary_path / f"run-36-{run_index}.json"
                        print(
                            f"\n=== Conditional sweep: 36 GPU layers, "
                            f"run {run_index}/{RUNS_PER_CONFIGURATION} ==="
                        )
                        report = _run_configuration(
                            "GPU layer sweep: 36",
                            36,
                            run_index,
                            json_path,
                        )
                        runs[36].append(report)
                        failures.extend(_validate_report(report, 36, run_index))

            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(
                _render_report(runs, failures, conditional_result, tested_next_layers),
                encoding="utf-8",
            )
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        failures.append(f"{type(error).__name__}: {error}")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            _render_report(runs, failures, conditional_result, tested_next_layers),
            encoding="utf-8",
        )
        print(f"GPU SWEEP ERROR: {type(error).__name__}: {error}")
    finally:
        if original_smoke_report is not None:
            SMOKE_REPORT_PATH.write_bytes(original_smoke_report)

    print(f"GPU SWEEP RESULT: {'PASS' if not failures else 'FAIL'}")
    print(f"REPORT: {REPORT_PATH.relative_to(ROOT)}")
    for failure in failures:
        print(f"FAILURE: {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
