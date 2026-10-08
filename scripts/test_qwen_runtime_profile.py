from __future__ import annotations

import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "test_qwen_video_smoke.py"
REPORT_PATH = ROOT / "data" / "evaluation" / "qwen_runtime_profile.md"
QUESTIONS = (
    "What happened in this video?",
    "What about the sports ball?",
    "When did it appear?",
    "What text was shown?",
)
RUNS_PER_CONFIGURATION = 2
CONFIGURATIONS = (
    ("Current: 24 GPU layers", 24),
    ("Test: 28 GPU layers", 28),
)
METRICS = (
    ("first_token_seconds", "First token (s)"),
    ("prompt_processing_estimate_seconds", "Prompt/prefill (s)"),
    ("completion_seconds", "Generation (s)"),
    ("total_response_seconds", "Total response (s)"),
    ("prompt_tokens", "Prompt tokens"),
    ("completion_tokens", "Completion tokens"),
    ("tokens_per_second", "Generation tokens/s"),
)


def _average(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _run_configuration(
    label: str,
    gpu_layers: int,
    run_index: int,
    json_path: Path,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["FRIDAY_QWEN_GPU_LAYERS"] = str(gpu_layers)
    environment["FRIDAY_QWEN_PROFILE_LABEL"] = label
    environment["FRIDAY_QWEN_PROFILE_JSON"] = str(json_path)
    completed = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if not json_path.is_file():
        raise RuntimeError(
            f"{label} run {run_index} produced no profile data "
            f"(exit {completed.returncode})."
        )
    report = json.loads(json_path.read_text(encoding="utf-8"))
    report["process_exit_code"] = completed.returncode
    report["run_index"] = run_index
    report["profile_label"] = label
    return report


def _turn_metric(
    reports: list[dict[str, Any]],
    question: str,
    metric_name: str,
) -> float | None:
    values = [
        float(turn[metric_name])
        for report in reports
        for turn in report.get("turns", [])
        if turn.get("question") == question
        and turn.get(metric_name) is not None
    ]
    return _average(values)


def _visual_mean(
    reports: list[dict[str, Any]],
    metric_name: str,
) -> float | None:
    return _average(
        [
            float(turn[metric_name])
            for report in reports
            for turn in report.get("turns", [])
            if turn.get("question") in QUESTIONS[:3]
            and turn.get(metric_name) is not None
        ]
    )


def _selected_device_is_rtx(report: dict[str, Any]) -> bool:
    runtime = report.get("runtime", {})
    selected = runtime.get("vulkan_device_selected")
    devices = runtime.get("vulkan_devices", "")
    return bool(
        isinstance(selected, str)
        and re.search(
            rf"^\s*{re.escape(selected)}\s*:\s*NVIDIA GeForce RTX 4050\b",
            str(devices),
            re.IGNORECASE | re.MULTILINE,
        )
    )


def _clean_log_text(value: object) -> str:
    return str(value or "").replace("\r", "").strip()


def _render_report(
    runs: dict[str, list[dict[str, Any]]],
    failures: list[str],
) -> str:
    current = runs[CONFIGURATIONS[0][0]]
    tested = runs[CONFIGURATIONS[1][0]]
    baseline_runtime = current[0].get("runtime", {})
    lines = [
        "# Qwen3-8B runtime performance profile",
        "",
        f"- Result: **{'PASS' if not failures else 'FAIL'}**",
        "- Device: NVIDIA GeForce RTX 4050 Laptop GPU; adapter selects its Vulkan device explicitly.",
        "- Other enumerated device: AMD Radeon 740M Graphics (`Vulkan0`); Qwen selected `Vulkan1`.",
        f"- llama.cpp executable: `{baseline_runtime.get('executable')}`",
        f"- llama.cpp build: `{_clean_log_text(baseline_runtime.get('version'))}`",
        f"- GPU: `{baseline_runtime.get('vulkan_device_selected')}`",
        f"- CPU: AMD Ryzen 7 7445HS (6 cores / 12 logical processors)",
        f"- CPU generation/prompt threads: {baseline_runtime.get('cpu_threads')}",
        "- Logical batch / physical ubatch: "
        f"{baseline_runtime.get('batch_size')} / {baseline_runtime.get('ubatch_size')}",
        f"- Context size: {baseline_runtime.get('context_size')}",
        f"- Repetitions per configuration: {RUNS_PER_CONFIGURATION}",
        f"- Flash attention: {baseline_runtime.get('flash_attention')}",
        f"- KV cache types: {baseline_runtime.get('kv_cache_types')}",
        f"- Other flags: `{baseline_runtime.get('other_launch_flags')}`",
        "",
        "## Runtime startup diagnostics",
        "",
        "```text",
        _clean_log_text(baseline_runtime.get("startup_diagnostics", "")),
        "```",
        "",
        "The measured first-token interval minus HTTP connection setup is an estimate of prompt/prefill "
        "time; llama-server does not expose an independent prefill timer in this adapter path.",
        "",
        "## Configuration comparison",
        "",
        "| Metric (mean over three visual turns and both runs) | Current (24 layers) | Test (28 layers) | Change |",
        "|---|---:|---:|---:|",
    ]
    metric_keys = (
        ("startup_seconds", "Model/server startup (s)"),
        ("prompt_processing_estimate_seconds", "Prompt/prefill (s)"),
        ("first_token_seconds", "First token (s)"),
        ("completion_seconds", "Generation (s)"),
        ("total_response_seconds", "Total FRIDAY turn (s)"),
        ("prompt_tokens", "Prompt tokens"),
        ("completion_tokens", "Completion tokens"),
        ("tokens_per_second", "Generation tokens/s"),
    )
    for key, label in metric_keys:
        if key == "startup_seconds":
            before = _average(
                [
                    float(report["runtime"]["server_model_load_seconds"])
                    for report in current
                    if report.get("runtime", {}).get("server_model_load_seconds")
                    is not None
                ]
            )
            after = _average(
                [
                    float(report["runtime"]["server_model_load_seconds"])
                    for report in tested
                    if report.get("runtime", {}).get("server_model_load_seconds")
                    is not None
                ]
            )
        else:
            before = _visual_mean(current, key)
            after = _visual_mean(tested, key)
        change = (
            f"{(after - before) / before * 100:+.1f}%"
            if before not in (None, 0) and after is not None
            else "n/a"
        )
        lines.append(
            f"| {label} | {before:.3f} | {after:.3f} | {change} |"
            if before is not None and after is not None
            else f"| {label} | {before} | {after} | {change} |"
        )

    lines.extend(
        [
            "",
            "## Same-turn comparison",
            "",
            "| Turn | Metric | Current (24 layers) | Test (28 layers) | Change |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for question in QUESTIONS:
        for metric_name, metric_label in METRICS:
            before = _turn_metric(current, question, metric_name)
            after = _turn_metric(tested, question, metric_name)
            change = (
                f"{(after - before) / before * 100:+.1f}%"
                if before not in (None, 0) and after is not None
                else "n/a"
            )
            lines.append(
                f"| {question} | {metric_label} | {before:.3f} | {after:.3f} | {change} |"
                if before is not None and after is not None
                else f"| {question} | {metric_label} | {before} | {after} | {change} |"
            )

    lines.extend(
        [
            "",
            "## Memory and GPU samples",
            "",
            "| Configuration | Peak process-tree RSS (MiB) | Peak RTX 4050 memory used (MiB) | Mean RTX 4050 utilization (%) | GPU samples |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, reports in (
        (CONFIGURATIONS[0][0], current),
        (CONFIGURATIONS[1][0], tested),
    ):
        rss_values = [
            float(report["memory"]["peak_process_tree_rss_mib"])
            for report in reports
            if report.get("memory", {}).get("peak_process_tree_rss_mib") is not None
        ]
        gpu = [
            report.get("memory", {}).get("nvidia", {})
            for report in reports
        ]
        gpu_peaks = [
            float(item["peak_memory_used_mib"])
            for item in gpu
            if item.get("peak_memory_used_mib") is not None
        ]
        gpu_util = [
            float(item["average_utilization_percent"])
            for item in gpu
            if item.get("average_utilization_percent") is not None
        ]
        sample_count = sum(int(item.get("sample_count", 0)) for item in gpu)
        lines.append(
            f"| {label} | {_average(rss_values):.1f} | "
            f"{max(gpu_peaks) if gpu_peaks else 'unavailable'} | "
            f"{_average(gpu_util):.1f} | {sample_count} |"
        )

    lines.extend(
        [
            "",
            "## Grounding and run validation",
            "",
        ]
    )
    for label, reports in (
        (CONFIGURATIONS[0][0], current),
        (CONFIGURATIONS[1][0], tested),
    ):
        for report in reports:
            all_checks = all(
                all(turn.get("checks", {}).values())
                and turn.get("first_token_seconds") is not None
                and turn.get("completion_seconds") is not None
                for turn in report.get("turns", [])
            )
            rtx_selected = _selected_device_is_rtx(report)
            lines.append(
                f"- {label}, run {report['run_index']}: smoke "
                f"{report['status']}; grounding checks {'PASS' if all_checks else 'FAIL'}; "
                f"RTX 4050 selected {'YES' if rtx_selected else 'NO'}."
            )
    lines.extend(["", "## Findings", ""])
    if current and tested:
        baseline_prefill = _visual_mean(current, "prompt_processing_estimate_seconds")
        tested_prefill = _visual_mean(tested, "prompt_processing_estimate_seconds")
        baseline_generation = _visual_mean(current, "completion_seconds")
        tested_generation = _visual_mean(tested, "completion_seconds")
        lines.append(
            "- Prefill remains the dominant visual-turn cost; compare its measured "
            "change with generation time and repeat-to-repeat spread before attributing "
            "a gain to the GPU-layer change."
        )
        lines.append(
            f"- Mean visual prefill: {baseline_prefill:.3f}s → {tested_prefill:.3f}s; "
            f"mean generation: {baseline_generation:.3f}s → {tested_generation:.3f}s."
        )
        lines.append(
            "- Same four user turns and grounding checks were used in every run. "
            "Only `--n-gpu-layers` changed (24 → 28); no prompt, model, context, "
            "batch, thread, cache, or FRIDAY behavior changes were made for this comparison."
            " Runs used a counterbalanced 24/28/28/24 order to reduce run-order/cache bias."
        )
    lines.extend(["", "## Failures", ""])
    lines.extend(f"- {failure}" for failure in failures) if failures else lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    runs = {label: [] for label, _ in CONFIGURATIONS}
    failures: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="friday-qwen-profile-") as temporary:
            temporary_path = Path(temporary)
            configuration_by_layers = {
                gpu_layers: label for label, gpu_layers in CONFIGURATIONS
            }
            run_order = (24, 28, 28, 24)
            counts = {label: 0 for label, _ in CONFIGURATIONS}
            for sequence_index, gpu_layers in enumerate(run_order, start=1):
                label = configuration_by_layers[gpu_layers]
                counts[label] += 1
                run_index = counts[label]
                json_path = temporary_path / f"profile-{sequence_index}.json"
                print(
                    f"\n=== Profile order {sequence_index}/4: {label}, "
                    f"run {run_index}/{RUNS_PER_CONFIGURATION} ==="
                )
                report = _run_configuration(
                    label,
                    gpu_layers,
                    run_index,
                    json_path,
                )
                runs[label].append(report)
                checks_pass = all(
                    all(turn.get("checks", {}).values())
                    and turn.get("first_token_seconds") is not None
                    and turn.get("completion_seconds") is not None
                    for turn in report.get("turns", [])
                )
                if (
                    report.get("status") != "PASS"
                    or not checks_pass
                    or report.get("process_exit_code") != 0
                ):
                    failures.append(
                        f"{label} run {run_index} failed smoke or grounding checks."
                    )
                if not report.get("runtime", {}).get("base_url_loopback"):
                    failures.append(
                        f"{label} run {run_index} did not use loopback."
                    )
                if not _selected_device_is_rtx(report):
                    failures.append(
                        f"{label} run {run_index} did not map selected device "
                        "to RTX 4050."
                    )
                if not report.get("memory", {}).get("nvidia", {}).get(
                    "sample_count"
                ):
                    failures.append(
                        f"{label} run {run_index} has no NVIDIA GPU samples."
                    )
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(
                _render_report(runs, failures),
                encoding="utf-8",
            )
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        failures.append(f"{type(error).__name__}: {error}")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            _render_report(runs, failures),
            encoding="utf-8",
        )
        print(f"PROFILE ERROR: {type(error).__name__}: {error}")

    print(f"PROFILE RESULT: {'PASS' if not failures else 'FAIL'}")
    print(f"REPORT: {REPORT_PATH.relative_to(ROOT)}")
    for failure in failures:
        print(f"FAILURE: {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
