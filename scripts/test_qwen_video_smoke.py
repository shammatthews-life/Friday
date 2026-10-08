from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.conversation_engine import ConversationEngine
from src.llm.qwen_adapter import QwenAdapter
from src.llm.video_context_bridge import VideoConversationBridge
from src.perception.pipeline import create_pipeline
from src.perception.video.context import VideoContextBuilder, VideoContextLimits
from src.perception.video.ocr import TesseractOCRBackend, VideoTextExtractor
from src.perception.video.question import VideoQuestionInterface
from src.perception.video.session import VideoAnalysisSession
from src.perception.video.video_source import VideoFrameSource


VIDEO_PATH = ROOT / "data" / "test_videos" / "friday_video_smoke.avi"
MODEL_PATH = ROOT / "models" / "llm" / "qwen3-8b" / "Qwen3-8B-Q4_K_M.gguf"
RUNTIME_DIR = ROOT / ".venv" / "llama.cpp"
REPORT_PATH = ROOT / "data" / "evaluation" / "qwen_video_smoke_test.md"
QUESTIONS = (
    "What happened in this video?",
    "What about the sports ball?",
    "When did it appear?",
    "What text was shown?",
)
INTERNAL_PATTERN = re.compile(
    r"\b(?:video\.question|full_video|visual_events|time_range|track[_ ]?id|"
    r"frame[_ ]?index|capability)\b",
    re.IGNORECASE,
)
DISTANCE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|meters?|metres?|km|kilometers?|"
    r"kilometres?|inches?|feet|ft|yards?|miles?)\b",
    re.IGNORECASE,
)
ABSOLUTE_NO_TEXT_PATTERN = re.compile(
    r"\b(?:no|not any)\s+(?:readable\s+|visible\s+)?text\b"
    r".{0,40}\b(?:shown|present|visible|exist(?:ed|s)?|observed|detected|found)\b"
    r"|\b(?:video|clip)\s+(?:contains?|shows?)\s+no text\b"
    r"|\bthere (?:is|was) no text\b",
    re.IGNORECASE,
)
OCR_LIMIT_ACKNOWLEDGEMENT_PATTERN = re.compile(
    r"\b(?:no text was recognized|text was not recognized|"
    r"OCR did not recognize|no recognized text|could not read|"
    r"couldn't read|not detected)\b",
    re.IGNORECASE,
)
TEXT_PRESENCE_UNCERTAINTY_PATTERN = re.compile(
    r"\b(?:whether|if)\b.{0,80}\b(?:text|writing|words?)\b"
    r".{0,80}\b(?:cannot|can't|could not)\s+"
    r"(?:be\s+)?(?:determined|determine|tell|know|establish|confirm)\b"
    r"|\b(?:cannot|can't|could not)\s+(?:be\s+)?"
    r"(?:determined|determine|tell|know|establish|confirm)\b"
    r".{0,80}\b(?:whether|if)\b.{0,80}\b(?:text|writing|words?)\b",
    re.IGNORECASE,
)


class ProcessTreeResourceMonitor:
    def __init__(self, interval_seconds: float = 0.25) -> None:
        import psutil

        self.psutil = psutil
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_rss_bytes = 0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
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
                self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied, OSError):
                pass
            self.stop_event.wait(self.interval_seconds)


class NvidiaResourceMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.samples: list[dict[str, float]] = []
        self.device_name: str | None = None
        self.errors: list[str] = []

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        fields = [item.strip() for item in line.split(",")]
                        if len(fields) != 4 or "RTX 4050" not in fields[0]:
                            continue
                        self.device_name = fields[0]
                        self.samples.append(
                            {
                                "memory_total_mib": float(fields[1]),
                                "memory_used_mib": float(fields[2]),
                                "utilization_percent": float(fields[3]),
                            }
                        )
                        break
            except (OSError, subprocess.SubprocessError, ValueError):
                if len(self.errors) < 3:
                    self.errors.append("Unable to sample NVIDIA GPU metrics.")
            self.stop_event.wait(self.interval_seconds)

    def summary(self) -> dict[str, object]:
        return {
            "device_name": self.device_name,
            "sample_count": len(self.samples),
            "sampling_errors": list(self.errors),
            "peak_memory_used_mib": max(
                (item["memory_used_mib"] for item in self.samples),
                default=None,
            ),
            "average_utilization_percent": (
                sum(item["utilization_percent"] for item in self.samples)
                / len(self.samples)
                if self.samples
                else None
            ),
            "peak_utilization_percent": max(
                (item["utilization_percent"] for item in self.samples),
                default=None,
            ),
        }


def _compact_context_interface() -> VideoQuestionInterface:
    limits = VideoContextLimits(
        max_objects=20,
        max_events=40,
        max_episodes=20,
        max_text_observations=40,
        max_text_history_episodes=20,
        max_text_history_events=40,
        max_evidence_references=40,
        max_nested_events=8,
        max_nested_evidence=8,
        max_nested_text_observations=8,
    )
    return VideoQuestionInterface(
        context_builder_factory=lambda knowledge: VideoContextBuilder(
            knowledge,
            limits=limits,
        )
    )


def _answer_checks(
    question: str,
    response: str,
    evidence_status: str,
    context: dict[str, object],
    previous_track_id: int | None,
) -> dict[str, bool]:
    context_events = context["visual_events"]
    context_objects = context["visual_objects"]
    text_observations = context["text_observations"]
    evidence_refs = context["evidence_references"]
    checks = {
        "nonempty_real_model_response": bool(response.strip()),
        "no_internal_identifiers": INTERNAL_PATTERN.search(response) is None,
        "no_invented_physical_measurements": DISTANCE_PATTERN.search(response) is None,
    }
    observation_status = context.get("text_observation_status")
    if (
        isinstance(observation_status, dict)
        and observation_status.get("status") == "no_recognized_observations"
    ):
        checks["no_absolute_text_absence_claim"] = (
            ABSOLUTE_NO_TEXT_PATTERN.search(response) is None
        )
    if question == QUESTIONS[0]:
        checks["full_video_context_supplied"] = context["selection"]["kind"] == "full_video"
        checks["visual_claims_have_supporting_events"] = (
            bool(context_events) or evidence_status == "insufficient_context"
        )
        supported_labels = {
            item["label"].casefold() for item in context_objects
        }
        response_lower = response.casefold()
        checks["visual_answer_uses_context_label"] = any(
            label in response_lower for label in supported_labels
        )
        checks["visual_events_have_matching_object_evidence"] = all(
            any(
                event["track_id"] == reference["track_id"]
                and event["label"] == reference["label"]
                and event["timestamp"] == reference["timestamp"]
                for event in context_events
            )
            for reference in evidence_refs
            if reference["label"].casefold() in supported_labels
        )
        checks["evidence_references_preserved"] = all(
            any(
                ref["timestamp"] == event["timestamp"]
                and ref["source_frame_index"] == event["frame_index"]
                and ref["track_id"] == event["track_id"]
                for ref in evidence_refs
            )
            for event in context_events
        )
    elif question == QUESTIONS[1]:
        sports_ball = [
            item
            for item in context_objects
            if item["label"].casefold() == "sports ball"
        ]
        checks["sports_ball_track_context_supplied"] = bool(sports_ball)
        checks["sports_ball_track_is_unambiguous"] = len(sports_ball) == 1
        checks["sports_ball_identity_not_guessed"] = (
            evidence_status == "insufficient_context" or len(sports_ball) == 1
        )
    elif question == QUESTIONS[2]:
        supplied_timestamps = {
            float(event["timestamp"])
            for event in context_events
            if event["timestamp"] is not None
        }
        response_numbers = {
            float(number)
            for number in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)", response)
        }
        checks["followup_resolved_to_existing_track"] = (
            previous_track_id is not None
            and context["selection"].get("track_id") == previous_track_id
        )
        checks["followup_events_belong_to_track"] = bool(context_events) and all(
            event["track_id"] == previous_track_id for event in context_events
        )
        checks["timestamp_answer_uses_supplied_event_timestamp"] = (
            bool(response_numbers)
            and response_numbers.issubset(supplied_timestamps)
        )
    elif question == QUESTIONS[3]:
        recognized = {
            " ".join(item["text"].casefold().split())
            for item in text_observations
            if item["text"].strip()
        }
        if recognized:
            checks["ocr_response_uses_only_recognized_text"] = any(
                text in " ".join(response.casefold().split())
                for text in recognized
            )
        else:
            observation_status = context.get("text_observation_status")
            checks["ocr_context_states_no_recognized_text"] = (
                not text_observations
                and isinstance(observation_status, dict)
                and observation_status.get("status") == "no_recognized_observations"
                and "does not establish" in observation_status.get(
                    "interpretation", ""
                )
            )
            checks["ocr_does_not_claim_video_has_no_text"] = (
                ABSOLUTE_NO_TEXT_PATTERN.search(response) is None
            )
            checks["ocr_acknowledges_recognition_limit"] = (
                OCR_LIMIT_ACKNOWLEDGEMENT_PATTERN.search(response) is not None
                and TEXT_PRESENCE_UNCERTAINTY_PATTERN.search(response) is not None
            )
        checks["ocr_context_contains_grounded_observations"] = all(
            item["text"].strip() and item["timestamp"] is not None
            for item in text_observations
        )
    return checks


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Real Qwen grounded video smoke test",
        "",
        f"- Result: **{report['status']}**",
        f"- Model: `{report['model']['path']}` ({report['model'].get('identity', 'identity not verified')})",
        f"- Runtime: `{report['runtime'].get('version', 'not started')}`",
        f"- Context size: {report['runtime'].get('context_size')}",
        f"- Verified local Qwen model identity: {report['runtime'].get('qwen_identity_verified', False)}",
        f"- Video: `{report['video']['path']}`",
        f"- Session result: `{report['video'].get('session_status', 'not run')}`",
        f"- Sampled/processed frames: {report['video'].get('sampled_frames', 0)}/{report['video'].get('processed_frames', 0)}",
        f"- Qwen startup: {report['timing'].get('qwen_startup_seconds')}",
        f"- Peak sampled process-tree RSS: {report['memory'].get('peak_process_tree_rss_mib')} MiB",
        "",
        "## OCR/text observations",
        "",
    ]
    observations = report["video"].get("text_observations", [])
    if observations:
        lines.extend(
            f"- {item.get('text', '')!r} at {item.get('timestamp')} "
            f"(frame {item.get('frame_index')})"
            for item in observations
        )
    else:
        lines.append("- No recognized text observations.")
    lines.extend(
        [
            "",
            "## FRIDAY responses and timings",
            "",
        ]
    )
    for turn in report["turns"]:
        lines.extend(
            [
                f"### {turn['question']}",
                "",
                f"- Response: {turn.get('response', '(no response)')}",
                f"- First token: {turn.get('first_token_seconds')}",
                f"- Prompt/context construction: {turn.get('context_construction_seconds')} / "
                f"{turn.get('prompt_construction_seconds')} seconds",
                f"- Request setup / connection setup: {turn.get('request_setup_seconds')} / "
                f"{turn.get('request_connection_setup_seconds')} seconds",
                f"- Estimated prompt processing: {turn.get('prompt_processing_estimate_seconds')} seconds",
                f"- Completion generation: {turn.get('completion_seconds')} seconds",
                f"- Request round-trip: {turn.get('request_round_trip_seconds')} seconds",
                f"- Total FRIDAY response: {turn.get('total_response_seconds')}",
                f"- Prompt/completion tokens: {turn.get('prompt_tokens')} / "
                f"{turn.get('completion_tokens')}",
                f"- Request bytes: {turn.get('request_bytes')}",
                "- Prompt message character counts: "
                + ", ".join(
                    f"{item['role']}={item['characters']}"
                    for item in turn.get("message_content_characters", [])
                ),
                "- Context section character counts: "
                + ", ".join(
                    f"{name}={characters}"
                    for name, characters in turn.get(
                        "context_section_characters", {}
                    ).items()
                ),
                f"- Context status: `{turn.get('evidence_status')}`",
                "- Checks:",
            ]
        )
        lines.extend(
            f"  - {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in turn.get("checks", {}).items()
        )
        if turn["question"] == QUESTIONS[3]:
            text_observations = turn.get("context", {}).get(
                "text_observations", []
            )
            lines.append("- Recognized text supplied in this turn's context:")
            if text_observations:
                lines.extend(
                    f"  - {item.get('text', '')!r} at {item.get('timestamp')} "
                    f"(frame {item.get('frame_index')})"
                    for item in text_observations
                )
            else:
                lines.append("  - None.")
        lines.append("")
    lines.extend(["## Failures", ""])
    failures = report.get("failures", [])
    lines.extend(f"- {failure}" for failure in failures) if failures else lines.append("- None")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Uses the existing local Qwen3-8B Q4_K_M adapter and its verified loopback llama.cpp server.",
            "- Uses the repository's short synthetic AVI, realtime YOLO26n profile, existing Tesseract backend, and existing video session/context path.",
            "- RSS is sampled process-tree resident memory using the existing `psutil` dependency; sampling is approximate.",
            "- Detector labels describe model outputs on synthetic geometry and are not ground-truth object labels.",
            "- No real-world or long-video inference was run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report: dict[str, Any] = {
        "status": "FAIL",
        "model": {"path": str(MODEL_PATH.relative_to(ROOT))},
        "runtime": {
            "qwen_identity_verified": False,
            "gpu_layers": int(
                os.environ.get("FRIDAY_QWEN_GPU_LAYERS", "24")
            ),
            "profile_label": os.environ.get("FRIDAY_QWEN_PROFILE_LABEL", "smoke"),
        },
        "video": {"path": str(VIDEO_PATH.relative_to(ROOT))},
        "timing": {},
        "memory": {},
        "turns": [],
        "failures": [],
    }
    adapter: QwenAdapter | None = None
    pipeline = None
    monitor: ProcessTreeResourceMonitor | None = None
    nvidia_monitor: NvidiaResourceMonitor | None = None
    all_checks_passed = False
    try:
        if not VIDEO_PATH.is_file():
            raise FileNotFoundError(f"Synthetic video is missing: {VIDEO_PATH}")
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Existing local Qwen model is missing: {MODEL_PATH}")

        monitor = ProcessTreeResourceMonitor()
        monitor.start()
        nvidia_monitor = NvidiaResourceMonitor()
        nvidia_monitor.start()
        adapter = QwenAdapter(
            model_path=MODEL_PATH,
            runtime_dir=RUNTIME_DIR,
            context_size=4096,
            gpu_layers=report["runtime"]["gpu_layers"],
            max_tokens=160,
            startup_timeout=240,
            request_timeout=300,
        )
        startup_started = time.perf_counter()
        adapter.start()
        report["timing"]["qwen_startup_seconds"] = time.perf_counter() - startup_started
        report["model"]["identity"] = "Qwen3-8B GGUF / Q4_K_M"
        report["runtime"].update(
            {
                "version": adapter.runtime_version,
                "device": adapter.vulkan_device,
                "context_size": adapter.context_size,
                "qwen_identity_verified": True,
                "base_url_loopback": adapter.base_url.startswith("http://127.0.0.1:"),
                "executable": str(adapter.runtime_executable),
                "vulkan_devices": adapter.runtime_devices,
                "vulkan_device_selected": adapter.vulkan_device,
                "gpu_layers": adapter.gpu_layers,
                "server_model_load_seconds": adapter.startup_seconds,
                "cpu_threads": adapter.runtime_threads,
                "batch_size": 128,
                "ubatch_size": 32,
                "context_size": adapter.context_size,
                "flash_attention": "auto (default; not specified)",
                "kv_cache_types": "default/auto (not specified)",
                "other_launch_flags": {
                    "parallel": 1,
                    "jinja": True,
                    "reasoning": "off",
                    "webui": False,
                    "device": adapter.vulkan_device,
                },
                "launch_command": adapter.runtime_command,
                "startup_diagnostics": adapter.startup_diagnostics,
            }
        )
        report["memory"]["nvidia"] = nvidia_monitor.summary()

        pipeline = create_pipeline("realtime", root=ROOT)
        source = VideoFrameSource(VIDEO_PATH, every_n_frames=2)
        text_extractor = VideoTextExtractor(TesseractOCRBackend())
        session_started = time.perf_counter()
        result = VideoAnalysisSession(
            source,
            pipeline,
            text_extractor=text_extractor,
            history_size=64,
        ).run()
        report["timing"]["video_session_seconds"] = time.perf_counter() - session_started
        report["video"].update(
            {
                "session_status": result.status,
                "sampled_frames": result.sampled_frame_count,
                "processed_frames": result.processed_frame_count,
                "decoded_frames": source.decoded_frames,
                "fps": source.fps,
                "detected_labels": sorted(
                    {item.label for item in result.facts.objects}
                ),
                "text_observations": [
                    item.to_dict() for item in result.text_observations
                ],
                "session_errors": list(result.errors),
            }
        )
        if result.status != "completed":
            report["failures"].append(
                f"Video session status was {result.status}: {result.errors}"
            )
        if source.last_error:
            report["failures"].append(f"Video decoder error: {source.last_error}")

        bridge = VideoConversationBridge(
            result.knowledge,
            question_interface=_compact_context_interface(),
        )
        context_construction_times: list[float] = []
        original_handle_question = bridge.handle_question

        def timed_handle_question(question, memory):
            context_started = time.perf_counter()
            response = original_handle_question(question, memory)
            context_construction_times.append(time.perf_counter() - context_started)
            return response

        bridge.handle_question = timed_handle_question
        engine = ConversationEngine(adapter, video_bridge=bridge)
        previous_track_id: int | None = None
        for question in QUESTIONS:
            metric_start = len(adapter.completion_metrics)
            turn_started = time.perf_counter()
            response = engine.process(question)
            total_response = time.perf_counter() - turn_started
            metrics = adapter.completion_metrics[metric_start:]
            metric = metrics[-1] if metrics else {}
            # The bridge returns the same selected data to FRIDAY; reconstruct for report checks.
            queried = bridge.question_interface.query(
                question,
                result.knowledge,
                referent_track_id=previous_track_id,
            )
            context = queried.context or {}
            if question == QUESTIONS[1] and queried.referent_track_id is not None:
                previous_track_id = queried.referent_track_id
            elif question == QUESTIONS[1] and queried.status.value == "insufficient_context":
                candidates = [
                    item
                    for item in result.facts.objects
                    if item.label.casefold() == "sports ball"
                ]
                if len(candidates) == 1:
                    previous_track_id = candidates[0].track_id

            checks = _answer_checks(
                question,
                response,
                queried.status.value,
                context,
                previous_track_id,
            )
            turn = {
                "question": question,
                "response": response,
                "evidence_status": queried.status.value,
                "context": context,
                "first_token_seconds": metric.get("first_token_latency_seconds"),
                "completion_seconds": metric.get("completion_generation_seconds"),
                "request_round_trip_seconds": metric.get("response_latency_seconds"),
                "context_construction_seconds": (
                    context_construction_times[-1]
                    if context_construction_times
                    else None
                ),
                "prompt_construction_seconds": metric.get(
                    "prompt_construction_seconds"
                ),
                "request_setup_seconds": metric.get("request_setup_seconds"),
                "request_connection_setup_seconds": metric.get(
                    "request_connection_setup_seconds"
                ),
                "prompt_processing_estimate_seconds": metric.get(
                    "prompt_processing_estimate_seconds"
                ),
                "completion_generation_seconds": metric.get(
                    "completion_generation_seconds"
                ),
                "prompt_tokens": metric.get("prompt_tokens"),
                "completion_tokens": metric.get("generated_tokens"),
                "tokens_per_second": metric.get("tokens_per_second"),
                "request_bytes": metric.get("request_bytes"),
                "message_content_characters": metric.get(
                    "message_content_characters", []
                ),
                "context_section_characters": {
                    name: len(json.dumps(value, ensure_ascii=False))
                    for name, value in context.items()
                },
                "total_response_seconds": total_response,
                "checks": checks,
            }
            report["turns"].append(turn)
            if not all(checks.values()):
                report["failures"].append(
                    f"{question}: failed checks "
                    + ", ".join(name for name, passed in checks.items() if not passed)
                )
            print(f"FRIDAY: {response}")
            print(
                "TIMING: "
                f"first_token={turn['first_token_seconds']}s "
                f"completion={turn['completion_seconds']}s "
                f"total={total_response:.3f}s"
            )

        report["timing"]["qwen_total_completion_seconds"] = sum(
            turn["completion_seconds"] or 0.0 for turn in report["turns"]
        )
        report["memory"]["peak_process_tree_rss_mib"] = (
            round(monitor.peak_rss_bytes / (1024 * 1024), 1)
            if monitor is not None and monitor.peak_rss_bytes
            else None
        )
        report["memory"]["nvidia"] = nvidia_monitor.summary()
        all_checks_passed = (
            result.status == "completed"
            and len(report["turns"]) == len(QUESTIONS)
            and all(
                all(turn["checks"].values())
                and turn["first_token_seconds"] is not None
                and turn["completion_seconds"] is not None
                for turn in report["turns"]
            )
            and report["runtime"]["qwen_identity_verified"]
            and report["runtime"]["base_url_loopback"]
            and not report["failures"]
        )
        report["status"] = "PASS" if all_checks_passed else "FAIL"
    except Exception as error:
        report["failures"].append(f"{type(error).__name__}: {error}")
        print(f"SMOKE TEST ERROR: {type(error).__name__}: {error}")
    finally:
        if pipeline is not None:
            pipeline.close()
        if adapter is not None:
            adapter.close()
        if monitor is not None:
            monitor.stop()
            report["memory"]["peak_process_tree_rss_mib"] = (
                round(monitor.peak_rss_bytes / (1024 * 1024), 1)
                if monitor.peak_rss_bytes
                else None
            )
        if nvidia_monitor is not None:
            nvidia_monitor.stop()
            report["memory"]["nvidia"] = nvidia_monitor.summary()
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(_render_report(report), encoding="utf-8")
        profile_json_path = os.environ.get("FRIDAY_QWEN_PROFILE_JSON")
        if profile_json_path:
            Path(profile_json_path).write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    print(f"SMOKE RESULT: {report['status']}")
    print(f"REPORT: {REPORT_PATH.relative_to(ROOT)}")
    if report["failures"]:
        for failure in report["failures"]:
            print(f"FAILURE: {failure}")
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
