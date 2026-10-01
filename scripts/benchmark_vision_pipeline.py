from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
JSON_REPORT = ROOT / "data/evaluation/vision_pipeline_benchmark.json"
MARKDOWN_REPORT = ROOT / "data/evaluation/vision_pipeline_benchmark.md"
DEFAULT_IMAGE = ROOT / "data/test_images/scene_common_objects.png"


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_process_rss_bytes: int | None = None
        self.minimum_available_ram_bytes: int | None = None
        try:
            import psutil
        except ImportError:
            self.psutil = None
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
                process = self.psutil.Process()
                rss = process.memory_info().rss
                self.peak_process_rss_bytes = max(self.peak_process_rss_bytes or 0, rss)
                available = self.psutil.virtual_memory().available
                self.minimum_available_ram_bytes = min(
                    self.minimum_available_ram_bytes or available, available
                )
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied, OSError):
                pass
            self.stop_event.wait(self.interval_seconds)


def gpu_memory() -> dict[str, Any] | None:
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
    fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
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


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "min_ms": None, "max_ms": None, "fps": None}
    mean = sum(values) / len(values)
    return {
        "count": len(values),
        "mean_ms": mean,
        "min_ms": min(values),
        "max_ms": max(values),
        "fps": 1000 / mean if mean else None,
    }


def run_camera_benchmark(pipeline, seconds: float, config: dict[str, Any]) -> dict[str, Any]:
    if seconds <= 0:
        return {"status": "NOT_RUN", "duration_seconds": 0, "frames": 0, "capture_fps": None, "pipeline_fps": None}
    from src.perception.camera import CameraConfig, CameraSource

    settings = config.get("camera", {})
    camera = CameraSource(
        CameraConfig(
            index=settings.get("index", 0),
            width=settings.get("width", 640),
            height=settings.get("height", 480),
            backend=settings.get("backend", "CAP_DSHOW"),
            warmup_seconds=settings.get("warmup_seconds", 1.0),
            minimum_brightness=settings.get("minimum_brightness", 8.0),
            minimum_contrast=settings.get("minimum_contrast", 1.0),
        )
    )
    capture_frames = 0
    pipeline_frames = 0
    pipeline_latencies: list[float] = []
    depth_latencies: list[float] = []
    camera_detection_counts: list[int] = []
    camera_confidences: list[float] = []
    camera_labels: dict[str, float] = {}
    capture_duration = 0.0
    pipeline_duration = 0.0
    try:
        warmup_frame = camera.warm_up()
        actual_resolution = [int(warmup_frame.image.shape[1]), int(warmup_frame.image.shape[0])]

        capture_started = time.perf_counter()
        capture_deadline = capture_started + seconds
        while time.perf_counter() < capture_deadline:
            frame = camera.read()
            if frame is not None:
                capture_frames += 1
        capture_duration = time.perf_counter() - capture_started

        pipeline_started = time.perf_counter()
        pipeline_deadline = pipeline_started + seconds
        while time.perf_counter() < pipeline_deadline:
            frame = camera.read()
            if frame is None:
                continue
            output = pipeline.process(frame)
            if output.snapshot.valid:
                pipeline_latencies.append(output.timing.total_ms)
                pipeline_frames += 1
                camera_detection_counts.append(len(output.snapshot.scene_state.objects))
                for obj in output.snapshot.scene_state.objects:
                    camera_confidences.append(obj.confidence)
                    camera_labels[obj.label] = max(
                        camera_labels.get(obj.label, 0.0), obj.confidence
                    )
                if output.timing.depth_ms > 0:
                    depth_latencies.append(output.timing.depth_ms)
        pipeline_duration = time.perf_counter() - pipeline_started
        return {
            "status": "PASS" if capture_frames and pipeline_frames else "FAIL",
            "backend": settings.get("backend", "CAP_DSHOW"),
            "requested_resolution": [settings.get("width", 640), settings.get("height", 480)],
            "actual_resolution": actual_resolution,
            "capture_duration_seconds": capture_duration,
            "raw_capture_frames": capture_frames,
            "capture_fps": capture_frames / capture_duration if capture_duration else None,
            "pipeline_duration_seconds": pipeline_duration,
            "pipeline_frames": pipeline_frames,
            "pipeline_fps": 1000 / (sum(pipeline_latencies) / len(pipeline_latencies)) if pipeline_latencies else None,
            "pipeline_latency_mean_ms": sum(pipeline_latencies) / len(pipeline_latencies) if pipeline_latencies else None,
            "depth_call_count": len(depth_latencies),
            "first_depth_call_ms": depth_latencies[0] if depth_latencies else None,
            "mean_depth_call_ms": sum(depth_latencies) / len(depth_latencies) if depth_latencies else None,
            "detection_count_mean": (
                sum(camera_detection_counts) / len(camera_detection_counts)
                if camera_detection_counts
                else None
            ),
            "detected_labels_best_confidence": camera_labels,
            "confidence_summary": {
                "count": len(camera_confidences),
                "mean": sum(camera_confidences) / len(camera_confidences) if camera_confidences else None,
                "minimum": min(camera_confidences) if camera_confidences else None,
                "maximum": max(camera_confidences) if camera_confidences else None,
            },
            "rejected_frame_reason": camera.last_rejection,
        }
    except Exception as error:
        return {
            "status": "FAIL",
            "backend": settings.get("backend", "CAP_DSHOW"),
            "requested_resolution": [settings.get("width", 640), settings.get("height", 480)],
            "capture_duration_seconds": capture_duration,
            "raw_capture_frames": capture_frames,
            "pipeline_duration_seconds": pipeline_duration,
            "pipeline_frames": pipeline_frames,
            "depth_call_count": len(depth_latencies),
            "capture_fps": None,
            "pipeline_fps": None,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        camera.close()


def report_paths(profile: str) -> tuple[Path, Path]:
    if profile == "balanced":
        return JSON_REPORT, MARKDOWN_REPORT
    return (
        JSON_REPORT.with_name(f"vision_pipeline_benchmark_{profile}.json"),
        MARKDOWN_REPORT.with_name(f"vision_pipeline_benchmark_{profile}.md"),
    )


def write_report(
    report: dict[str, Any], json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    balanced = report["controlled_image"]
    camera = report["camera"]
    lines = [
        "# VisionAid Perception Pipeline Benchmark",
        "",
        "Controlled local image benchmark; smoke-test measurements, not an accuracy benchmark.",
        "",
        f"- Status: **{report['status']}**",
        f"- Profile: `{report['profile']}`",
        f"- Detector: `{report['detector']['name']}` (`{report['detector']['path']}`)",
        f"- Device: `{report['detector']['device']}`",
        f"- Model variants actually tested: {', '.join(report['model_variants_tested'])}",
        f"- Stronger local detector checkpoint: {report['stronger_model_status']}",
        "",
        "## Controlled Image Measurements",
        "",
        f"- Input: `{report['input_image']}`",
        f"- Model load: {report['detector']['load_time_ms']} ms",
        f"- Initial frame (cold pipeline): {balanced['warmup_pipeline_ms']} ms; scheduled depth: {balanced['warmup_depth_ms']} ms",
        f"- First scheduled depth warm-up: {report.get('depth_warmup')}",
        f"- Detector latency: {balanced['detector']['mean_ms']} ms mean; {balanced['detector']['fps']} FPS",
        f"- Detection count: mean {balanced['detection_count_mean']}; confidences: {balanced['confidence_summary']}",
        f"- Depth load: {report['depth']['load_time_ms']} ms",
        f"- Depth latency: {balanced['depth']['mean_ms']} ms mean",
        f"- First scheduled depth call: {balanced['first_scheduled_depth_ms']} ms",
        f"- Tracking overhead: {balanced['tracking']['mean_ms']} ms mean",
        f"- Fusion overhead: {balanced['fusion']['mean_ms']} ms mean",
        f"- Complete pipeline: {balanced['pipeline']['mean_ms']} ms mean; {balanced['pipeline']['fps']} FPS",
        "",
        "## Camera Measurements",
        "",
        f"- Status: {camera['status']}",
        f"- Backend/resolution: {camera.get('backend')} / requested {camera.get('requested_resolution')}, actual {camera.get('actual_resolution')}",
        f"- Raw capture: {camera.get('raw_capture_frames')} frames in {camera.get('capture_duration_seconds')} s; {camera.get('capture_fps')} FPS",
        f"- Pipeline: {camera.get('pipeline_frames')} frames in {camera.get('pipeline_duration_seconds')} s; {camera.get('pipeline_fps')} FPS",
        f"- Depth calls during camera pipeline: {camera.get('depth_call_count')}; first {camera.get('first_depth_call_ms')} ms; mean {camera.get('mean_depth_call_ms')} ms",
        f"- Camera detections: mean count {camera.get('detection_count_mean')}; best confidences {camera.get('detected_labels_best_confidence')}",
        "",
        "## Memory",
        "",
        f"- Peak process RSS: {report['memory']['peak_process_rss_bytes']} bytes",
        f"- Minimum available system RAM: {report['memory']['minimum_available_ram_bytes']} bytes",
        f"- GPU memory samples: {json.dumps(report['memory']['gpu'], ensure_ascii=False)}",
        "",
        "## Model Compatibility and Limits",
        "",
        "- YOLO26n is the only locally staged compatible closed-set detector tested by this benchmark.",
        "- YOLO26s is not locally staged; it was not downloaded or tested.",
        "- YOLOE-26n-seg remains an incompatible TorchScript archive for the installed Ultralytics loader; it was not replaced or modified.",
        "- Depth Anything V2 Small reports scene-relative depth only, never physical distance.",
        "- CUDA is unavailable to PyTorch in this environment; YOLO and depth measurements use CPU.",
        "- Synthetic/local image results do not establish real-world accuracy or certified safety.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the local VisionAid perception pipeline")
    parser.add_argument("--profile", default="balanced", choices=("quality", "balanced", "realtime"))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--camera-seconds", type=float, default=0.0)
    args = parser.parse_args()

    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import cv2
    import yaml

    config = yaml.safe_load((ROOT / "configs/perception.yaml").read_text(encoding="utf-8"))
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (ROOT / image_path).resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read controlled benchmark image: {image_path}")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")

    from src.perception.pipeline import create_pipeline
    from src.perception.types import Detection, PerceptionFrame

    sampler = ResourceSampler()
    gpu_before = gpu_memory()
    rss_before = sampler.psutil.Process().memory_info().rss if sampler.psutil else None
    sampler.start()
    pipeline = create_pipeline(args.profile, root=ROOT)
    detector = pipeline.detector
    depth_provider = pipeline.depth_provider
    try:
        warmup = pipeline.process(PerceptionFrame(image.copy(), source_id="controlled-image:warmup", frame_index=0))
        if not warmup.snapshot.valid:
            raise RuntimeError(f"Warm-up pipeline rejected the input: {warmup.snapshot.error}")
        records: list[dict[str, Any]] = []
        detection_counts: list[int] = []
        confidence_values: list[float] = []
        for index in range(args.iterations):
            output = pipeline.process(
                PerceptionFrame(
                    image=image.copy(),
                    timestamp=time.time(),
                    source_id="controlled-image",
                    frame_index=index + 1,
                )
            )
            if not output.snapshot.valid:
                raise RuntimeError(f"Pipeline rejected benchmark frame: {output.snapshot.error}")
            records.append(
                {
                    "detector_ms": output.timing.detection_ms,
                    "tracking_ms": output.timing.tracking_ms,
                    "depth_ms": output.timing.depth_ms,
                    "fusion_ms": output.timing.fusion_ms,
                    "pipeline_ms": output.timing.total_ms,
                }
            )
            detection_counts.append(len(output.snapshot.scene_state.objects))
            confidence_values.extend(obj.confidence for obj in output.snapshot.scene_state.objects)

        def average(key: str) -> list[float]:
            return [item[key] for item in records if item[key] > 0]

        depth_calls = average("depth_ms")
        controlled = {
            "iterations": args.iterations,
            "warmup_pipeline_ms": warmup.timing.total_ms,
            "warmup_detector_ms": warmup.timing.detection_ms,
            "warmup_depth_ms": warmup.timing.depth_ms,
            "warmup_tracking_ms": warmup.timing.tracking_ms,
            "warmup_fusion_ms": warmup.timing.fusion_ms,
            "detector": summarize(average("detector_ms")),
            "tracking": summarize(average("tracking_ms")),
            "depth": summarize(depth_calls),
            "first_scheduled_depth_ms": depth_calls[0] if depth_calls else None,
            "fusion": summarize(average("fusion_ms")),
            "pipeline": summarize(average("pipeline_ms")),
            "detection_count_mean": sum(detection_counts) / len(detection_counts),
            "confidence_summary": {
                "count": len(confidence_values),
                "mean": sum(confidence_values) / len(confidence_values) if confidence_values else None,
                "minimum": min(confidence_values) if confidence_values else None,
                "maximum": max(confidence_values) if confidence_values else None,
            },
        }
        depth_warmup = {"status": "not_enabled"}
        if depth_provider is not None and controlled["depth"]["count"] > 0:
            depth_warmup = {"status": "measured_in_controlled_iterations"}
        elif depth_provider is not None:
            height, width = image.shape[:2]
            full_frame_detection = Detection(
                label="benchmark_warmup_roi",
                confidence=1.0,
                bounding_box=(0.0, 0.0, float(width), float(height)),
                center_x=width / 2,
                center_y=height / 2,
                normalized_horizontal=0.5,
                normalized_vertical=0.5,
                position_category="center",
                vertical_position="middle",
                track_id=-1,
            )
            depth_started = time.perf_counter()
            try:
                depth_provider.estimate(
                    PerceptionFrame(
                        image=image.copy(),
                        timestamp=time.time(),
                        source_id="controlled-image:depth-warmup",
                        frame_index=None,
                    ),
                    [full_frame_detection],
                )
                depth_warmup = {
                    "status": "whole_frame_depth_warmup_only",
                    "inference_ms": (time.perf_counter() - depth_started) * 1000,
                    "not_a_detection": True,
                }
            except Exception as error:
                depth_warmup = {
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                }
        camera_result = run_camera_benchmark(pipeline, args.camera_seconds, config)
    finally:
        pipeline.close()
        sampler.stop()

    camera_pass = camera_result["status"] in {"PASS", "NOT_RUN"}
    try:
        import torch

        torch_cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        torch_cuda_available = None
    depth_warmup_pass = depth_warmup.get("status") != "FAIL"
    report = {
        "status": "PASS" if controlled["pipeline"]["count"] == args.iterations and camera_pass and depth_warmup_pass else "FAIL",
        "profile": args.profile,
        "input_image": str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path),
        "model_variants_tested": [config["profiles"][args.profile]["model_name"]],
        "stronger_model_status": "YOLO26s not locally staged; not downloaded or tested",
        "detector": {
            "name": config["profiles"][args.profile]["model_name"],
            "path": config["profiles"][args.profile]["model_path"],
            "device": config["profiles"][args.profile]["device"],
            "image_size": config["profiles"][args.profile]["image_size"],
            "load_time_ms": getattr(detector, "load_time_ms", None),
        },
        "depth": {
            "name": "Depth Anything V2 Small" if depth_provider else None,
            "relative_only": True,
            "load_time_ms": getattr(depth_provider, "load_time_ms", None),
        },
        "controlled_image": controlled,
        "depth_warmup": depth_warmup,
        "camera": camera_result,
        "memory": {
            "process_rss_before_bytes": rss_before,
            "peak_process_rss_bytes": sampler.peak_process_rss_bytes,
            "minimum_available_ram_bytes": sampler.minimum_available_ram_bytes,
            "gpu": {"before": gpu_before, "after": gpu_memory()},
        },
        "torch_cuda_available": torch_cuda_available,
        "network_model_downloads": False,
    }
    json_report, markdown_report = report_paths(args.profile)
    write_report(report, json_report, markdown_report)
    print(f"BENCHMARK STATUS: {report['status']}")
    print(f"DETECTOR MEAN MS: {controlled['detector']['mean_ms']}")
    print(f"DEPTH MEAN MS: {controlled['depth']['mean_ms']}")
    print(f"PIPELINE MEAN MS: {controlled['pipeline']['mean_ms']}")
    print(f"CAMERA CAPTURE FPS: {camera_result.get('capture_fps')}")
    print(f"PEAK PROCESS RAM BYTES: {sampler.peak_process_rss_bytes}")
    print(f"JSON REPORT: {json_report}")
    print(f"MARKDOWN REPORT: {markdown_report}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
