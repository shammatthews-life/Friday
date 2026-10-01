from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT_JSON = ROOT / "data/evaluation/qwen_friday_integration.json"
REPORT_MARKDOWN = ROOT / "data/evaluation/qwen_friday_integration.md"
MODEL_PATH = ROOT / "models/llm/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"
RUNTIME_DIR = ROOT / ".venv/llama.cpp"

from src.core.scene_memory import SceneMemory
from src.core.scene_state import SceneObject, SceneState
from src.llm.conversation_engine import ConversationEngine
from src.llm.qwen_adapter import QwenAdapter
from src.llm.tool_interface import (
    CapabilityKind,
    CapabilityRegistry,
    CapabilityRequest,
    CapabilityResult,
    GroundingData,
)
from src.llm.visionaid_bridge import (
    OBJECT_SEARCH,
    RELATIVE_DEPTH,
    SCENE_AWARENESS,
    VisionAidBridge,
)


class MockDepthProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, scene_object: SceneObject, image_path) -> dict[str, Any]:
        self.calls.append(scene_object.label)
        return {"category": "relatively near", "available": True}


class RecordingVisionAidBridge(VisionAidBridge):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict[str, Any]] = []

    def provide(self, request: CapabilityRequest) -> CapabilityResult:
        result = super().provide(request)
        self.calls.append(
            {
                "capability": request.capability,
                "arguments": dict(request.arguments),
                "available": result.available,
                "data": dict(result.data),
            }
        )
        return result


class MockSafetyProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def provide(self, request: CapabilityRequest) -> CapabilityResult:
        result = CapabilityResult(
            available=True,
            data={"state": "UNKNOWN", "known": False},
            grounding=GroundingData(safety_state={"state": "UNKNOWN", "known": False}),
        )
        self.calls.append({"capability": request.capability, "data": dict(result.data)})
        return result


def make_scene() -> SceneState:
    return SceneState(
        objects=[
            SceneObject("bottle", 0.84, position_category="left"),
            SceneObject("chair", 0.88, position_category="center"),
        ],
        timestamp=time.time(),
    )


def make_engine(adapter: QwenAdapter, scene: SceneState | None):
    scene_memory = SceneMemory()
    if scene is not None:
        scene_memory.update(scene)
    depth_provider = MockDepthProvider()
    bridge = RecordingVisionAidBridge(
        scene_provider=lambda: scene,
        scene_memory=scene_memory,
        depth_provider=depth_provider,
    )
    safety_provider = MockSafetyProvider()
    registry = bridge.register(CapabilityRegistry())
    registry.register("safety.proximity", safety_provider)
    return ConversationEngine(adapter, registry), bridge, safety_provider, depth_provider


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_process_tree_rss_bytes: int | None = None
        self.minimum_system_available_bytes: int | None = None
        self.error: str | None = None
        try:
            import psutil
        except ImportError:
            self.psutil = None
            self.error = "psutil unavailable; RAM sampling skipped"
        else:
            self.psutil = psutil

    def start(self) -> None:
        if self.psutil is not None:
            self.thread.start()

    def stop(self) -> None:
        if self.psutil is not None:
            self.stop_event.set()
            self.thread.join(timeout=3)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                parent = self.psutil.Process(os.getpid())
                processes = [parent, *parent.children(recursive=True)]
                rss = sum(
                    process.memory_info().rss
                    for process in processes
                    if process.is_running()
                )
                self.peak_process_tree_rss_bytes = max(
                    self.peak_process_tree_rss_bytes or 0, rss
                )
                available = self.psutil.virtual_memory().available
                self.minimum_system_available_bytes = min(
                    self.minimum_system_available_bytes or available, available
                )
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied, OSError):
                pass
            self.stop_event.wait(self.interval_seconds)


def nvidia_memory() -> dict[str, Any] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    result = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    if len(fields) != 4:
        return None
    try:
        return {
            "name": fields[0],
            "total_mib": int(fields[1]),
            "used_mib": int(fields[2]),
            "free_mib": int(fields[3]),
        }
    except ValueError:
        return None


def has_uncertainty(text: str) -> bool:
    return bool(
        re.search(
            r"\b(can't|cannot|unable|uncertain|unknown|unavailable|not enough|can't tell|don't know|do not know|can't determine|cannot determine)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def contains_metric_distance(text: str) -> bool:
    return bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(?:meters?|metres?|feet|foot|ft)\b", text, re.I)
    )


def response_hides_internals(text: str) -> bool:
    markers = (
        "vision.scene_awareness",
        "vision.object_search",
        "vision.relative_depth",
        "safety.proximity",
        "capabilityrequest",
        "groundingdata",
        "json",
        "tool call",
    )
    return not any(marker in text.lower() for marker in markers) and "{" not in text and "}" not in text


def write_reports(report: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# FRIDAY Qwen Conversation Integration",
        "",
        f"- Status: **{report.get('status', 'FAIL')}**",
        f"- Model: `{report['model']['path']}`",
        f"- Quantization: {report['model']['quantization']}",
        f"- Runtime: {report['runtime']['name']} ({report['runtime'].get('version') or 'unknown'})",
        f"- Backend/device: {report['runtime'].get('backend') or 'unknown'}",
        f"- Startup/load time: {report.get('startup_load_seconds')}",
        f"- Mean first-token latency: {report.get('mean_first_token_latency_seconds')}",
        f"- Mean total completion latency: {report.get('mean_completion_latency_seconds')}",
        f"- Generated tokens: {report.get('generated_tokens_total')}",
        f"- Aggregate generation speed: {report.get('tokens_per_second')}",
        f"- Peak process-tree RAM bytes: {report['memory'].get('peak_process_tree_rss_bytes')}",
        f"- Peak system GPU usage MiB: {report['memory'].get('gpu_peak_used_mib')}",
        "",
        "## Conversation",
        "",
    ]
    for turn in report.get("conversation", []):
        lines.extend(
            [
                f"### Turn {turn['turn']}",
                "",
                f"User: {turn['user']}",
                "",
                f"FRIDAY: {turn.get('response', '(no response)')}",
                "",
            ]
        )
    lines.extend(["## Grounding checks", ""])
    for name, value in report.get("checks", {}).items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", "## Capability calls", "", "```json"])
    lines.append(json.dumps(report.get("capability_calls", []), indent=2, ensure_ascii=False))
    lines.extend(["```", "", "## Errors", ""])
    lines.extend(f"- {error}" for error in report.get("errors", []))
    if not report.get("errors"):
        lines.append("- None")
    REPORT_MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_conversation(adapter: QwenAdapter, report: dict[str, Any]) -> None:
    no_scene_engine, no_scene_bridge, _, _ = make_engine(adapter, None)
    start_index = len(adapter.completion_metrics)
    no_scene_response = no_scene_engine.process("What do you see right now?")
    no_scene_calls = no_scene_bridge.calls
    report["grounding_scenarios"] = {
        "no_scene": {
            "user": "What do you see right now?",
            "response": no_scene_response,
            "capability_called": bool(no_scene_calls),
            "scene_available": no_scene_calls[-1]["data"].get("scene_available") if no_scene_calls else None,
            "completion_metrics": adapter.completion_metrics[start_index:],
        }
    }
    report["checks"]["no_scene_no_visual_invention"] = bool(
        no_scene_calls
        and no_scene_calls[-1]["data"].get("scene_available") is False
        and has_uncertainty(no_scene_response)
        and not re.search(r"\b(bottle|chair|person|table|door)\b", no_scene_response, re.I)
    )
    if not response_hides_internals(no_scene_response):
        report["checks"]["internal_identifiers_hidden"] = False

    scene = make_scene()
    engine, bridge, safety_provider, depth_provider = make_engine(adapter, scene)
    turns = [
        "Hey Friday, introduce yourself.",
        "I'm exhausted from studying today.",
        "Can you help me find my bottle?",
        "How far is it?",
        "Can you find my chair instead?",
        "Tell me a joke.",
        "Is there something dangerous around me?",
    ]
    responses: list[str] = []
    for index, text in enumerate(turns, start=1):
        call_count_before = len(bridge.calls) + len(safety_provider.calls)
        metric_start = len(adapter.completion_metrics)
        turn_started = time.perf_counter()
        response = engine.process(text)
        turn_elapsed = time.perf_counter() - turn_started
        responses.append(response)
        calls_after = len(bridge.calls) + len(safety_provider.calls)
        report["conversation"].append(
            {
                "turn": index,
                "user": text,
                "response": response,
                "total_turn_latency_seconds": turn_elapsed,
                "completion_metrics": adapter.completion_metrics[metric_start:],
                "capability_call_count": calls_after - call_count_before,
                "current_referent": engine.memory.current_referent,
            }
        )
        print(f"TURN {index} USER: {text}")
        print(f"TURN {index} FRIDAY: {response}")

    calls = bridge.calls
    safety_calls = safety_provider.calls
    bottle_search = next(
        (
            item
            for item in calls
            if item["capability"] == OBJECT_SEARCH
            and str(item["arguments"].get("target", "")).lower() == "bottle"
        ),
        None,
    )
    depth_call = next(
        (item for item in calls if item["capability"] == RELATIVE_DEPTH), None
    )
    chair_search = next(
        (
            item
            for item in calls
            if item["capability"] == OBJECT_SEARCH
            and str(item["arguments"].get("target", "")).lower() == "chair"
        ),
        None,
    )
    safety_response = responses[-1]
    joke_calls = report["conversation"][5]["capability_call_count"]
    all_user_replies = [no_scene_response, *responses]
    report["capability_calls"] = [*calls, *safety_calls]
    report["checks"].update(
        {
            "natural_introduction": bool(responses[0].strip()) and response_hides_internals(responses[0]),
            "empathetic_casual_reply": bool(
                re.search(r"(sorry|rough|tough|understand|exhaust|tiring|rest|break|sounds like)", responses[1], re.I)
            ),
            "casual_turn_no_capability": report["conversation"][1]["capability_call_count"] == 0,
            "bottle_search_grounded": bool(
                bottle_search
                and bottle_search["data"].get("found") is True
                and abs(float(bottle_search["data"].get("confidence") or 0) - 0.84) < 0.001
                and "bottle" in responses[2].lower()
                and re.search(r"\bleft\b", responses[2], re.I)
            ),
            "followup_context_resolved_to_bottle": bool(
                depth_call and str(depth_call["arguments"].get("target", "")).lower() == "bottle"
            ),
            "relative_depth_grounded": bool(
                depth_call
                and depth_call["data"].get("category") == "relatively near"
                and re.search(r"(relatively near|near|close)", responses[3], re.I)
                and not contains_metric_distance(responses[3])
            ),
            "chair_switch_and_search": bool(
                chair_search
                and str(chair_search["arguments"].get("target", "")).lower() == "chair"
                and chair_search["data"].get("found") is True
                and abs(float(chair_search["data"].get("confidence") or 0) - 0.88) < 0.001
                and report["conversation"][4]["current_referent"] == "chair"
                and engine.memory.current_referent == "chair"
            ),
            "joke_without_capability": bool(
                re.search(r"(why|joke|laugh|because)", responses[5], re.I)
                and joke_calls == 0
            ),
            "safety_unknown_grounded": bool(
                safety_calls
                and safety_calls[-1]["data"].get("state") == "UNKNOWN"
                and has_uncertainty(safety_response)
                and not re.search(r"\b(you are safe|you're safe|it is safe|it's safe|you are in danger|there is no danger)\b", safety_response, re.I)
            ),
            "internal_identifiers_hidden": all(
                response_hides_internals(reply) for reply in all_user_replies
            ),
        }
    )
    report["conversation_context"] = {
        "bottle_search_target": bottle_search["arguments"].get("target") if bottle_search else None,
        "depth_followup_target": depth_call["arguments"].get("target") if depth_call else None,
        "chair_search_target": chair_search["arguments"].get("target") if chair_search else None,
        "final_referent": engine.memory.current_referent,
    }
    report["capability_calls"] = [
        {"capability": call["capability"], "arguments": call.get("arguments", {}), "data": call.get("data", {})}
        for call in [*calls, *safety_calls]
    ]


def main() -> int:
    report: dict[str, Any] = {
        "status": "FAIL",
        "model": {
            "name": "Qwen3-8B",
            "path": str(MODEL_PATH.relative_to(ROOT)),
            "quantization": "Q4_K_M",
        },
        "runtime": {
            "name": "llama.cpp llama-server",
            "backend": "Vulkan / NVIDIA RTX 4050 (explicit device selection)",
            "bind_policy": "127.0.0.1 only",
            "offline": True,
            "context_size": 2048,
        },
        "memory": {},
        "conversation": [],
        "grounding_scenarios": {},
        "capability_calls": [],
        "checks": {},
        "errors": [],
    }
    adapter = QwenAdapter(
        model_path=MODEL_PATH,
        runtime_dir=RUNTIME_DIR,
        context_size=2048,
        gpu_layers=24,
        max_tokens=128,
        startup_timeout=240,
        request_timeout=300,
    )
    monitor = ResourceMonitor()
    gpu_before = nvidia_memory()
    total_started = time.perf_counter()
    monitor.start()
    try:
        report["startup_load_seconds"] = adapter.start()
        report["runtime"]["version"] = adapter.runtime_version
        report["runtime"]["vulkan_device"] = adapter.vulkan_device
        report["runtime"]["server_url"] = adapter.base_url
        report["runtime"]["server_owned_by_test"] = adapter.owns_server
        run_conversation(adapter, report)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        monitor.stop()
        adapter.close()
        report["memory"] = {
            "measurement": "psutil process-tree RSS sampled every 0.5 seconds",
            "peak_process_tree_rss_bytes": monitor.peak_process_tree_rss_bytes,
            "minimum_system_available_bytes": monitor.minimum_system_available_bytes,
            "sampling_note": monitor.error,
            "gpu_before": gpu_before,
            "gpu_after": nvidia_memory(),
            "gpu_peak_used_mib": None,
        }
        report["elapsed_seconds_total"] = time.perf_counter() - total_started
        metrics = adapter.completion_metrics
        first_tokens = [
            float(item["first_token_latency_seconds"])
            for item in metrics
            if item.get("first_token_latency_seconds") is not None
        ]
        latencies = [float(item["response_latency_seconds"]) for item in metrics]
        token_counts = [item.get("generated_tokens") for item in metrics]
        report["first_token_latency_seconds"] = first_tokens
        report["mean_first_token_latency_seconds"] = (
            sum(first_tokens) / len(first_tokens) if first_tokens else None
        )
        report["mean_completion_latency_seconds"] = (
            sum(latencies) / len(latencies) if latencies else None
        )
        report["generated_tokens_total"] = (
            sum(int(item) for item in token_counts)
            if token_counts and all(item is not None for item in token_counts)
            else None
        )
        report["tokens_per_second"] = (
            report["generated_tokens_total"] / sum(latencies)
            if report["generated_tokens_total"] is not None and sum(latencies) > 0
            else None
        )
        report["completion_count"] = len(metrics)
        check_values = list(report["checks"].values())
        report["status"] = (
            "PASS"
            if not report["errors"] and check_values and all(check_values)
            else "FAIL"
        )
        write_reports(report)

    print(f"QWEN FRIDAY INTEGRATION: {report['status']}")
    print(f"STARTUP/LOAD SECONDS: {report.get('startup_load_seconds')}")
    print(f"MEAN FIRST-TOKEN SECONDS: {report.get('mean_first_token_latency_seconds')}")
    print(f"TOKENS PER SECOND: {report.get('tokens_per_second')}")
    print(f"PEAK PROCESS-TREE RAM BYTES: {report['memory'].get('peak_process_tree_rss_bytes')}")
    print(f"REPORT JSON: {REPORT_JSON}")
    print(f"REPORT MARKDOWN: {REPORT_MARKDOWN}")
    if report["errors"]:
        print("ERRORS: " + " | ".join(report["errors"]))
    for name, passed in report["checks"].items():
        print(f"{name.upper()} = {'PASS' if passed else 'FAIL'}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
