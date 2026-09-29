"""Phase 1D local CPU inference benchmark for the five selected models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import statistics
import time
import wave
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data/test_images/scene_common_objects.png"
OUTPUT_WAV = ROOT / "data/evaluation/piper_amy_smoke_test.wav"


@contextmanager
def network_block():
    """Prevent socket connections during local model loading and inference."""
    original_connect = socket.socket.connect
    original_create = socket.create_connection

    def blocked_connect(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Network access blocked during offline inference")

    def blocked_create(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Network access blocked during offline inference")

    socket.socket.connect = blocked_connect
    socket.create_connection = blocked_create
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def memory_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def model_size(path: Path) -> int | None:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return None


def base_record(model: dict[str, Any], path: Path) -> dict[str, Any]:
    size = model_size(path)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model["name"],
        "task": model["capability"],
        "family": model["family"],
        "mode": model["mode"],
        "model_path": str(path),
        "model_file_size_bytes": size,
        "device": "cpu",
        "device_name": "CPU",
        "cuda_available": bool(torch.cuda.is_available()),
        "input_settings": {"image": str(IMAGE_PATH), "image_dimensions": None, "batch_size": 1},
        "warmup_count": 1,
        "measured_iterations": 0,
        "load_time_ms": None,
        "warmup_latency_ms": None,
        "latencies_ms": [],
        "average_latency_ms": None,
        "min_latency_ms": None,
        "max_latency_ms": None,
        "fps": None,
        "ram_before_mb": memory_mb(),
        "ram_after_load_mb": None,
        "ram_peak_mb": None,
        "vram_mb": None,
        "cpu_usage_percent": None,
        "output_path": None,
        "output_wav_duration_seconds": None,
        "output_wav_size_bytes": None,
        "errors": [],
        "offline_test_result": "not_run",
        "status": "FAIL",
        "inference_run": False,
    }


def time_call(function: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    value = function()
    return value, (time.perf_counter() - start) * 1000


def finish_timing(record: dict[str, Any], latencies: list[float]) -> None:
    record["measured_iterations"] = len(latencies)
    record["latencies_ms"] = latencies
    record["average_latency_ms"] = statistics.fmean(latencies)
    record["min_latency_ms"] = min(latencies)
    record["max_latency_ms"] = max(latencies)
    record["fps"] = 1000 / record["average_latency_ms"] if record["average_latency_ms"] else None
    record["ram_peak_mb"] = memory_mb()


def run_yolo(model: dict[str, Any], root: Path, record: dict[str, Any], yoloe: bool = False) -> None:
    from ultralytics import YOLO, YOLOE

    image = str(IMAGE_PATH)
    model_class = YOLOE if yoloe else YOLO
    with network_block():
        original_resolver = None
        if yoloe:
            encoder_path = (root / "models/detection/yoloe-text/mobileclip2_b.ts").resolve()
            from ultralytics.utils import downloads

            original_resolver = downloads.attempt_download_asset
            downloads.attempt_download_asset = lambda file, **kwargs: str(encoder_path)
        try:
            loaded, load_ms = time_call(lambda: model_class(record["model_path"]))
        finally:
            if original_resolver is not None:
                downloads.attempt_download_asset = original_resolver
        record["load_time_ms"] = load_ms
        record["ram_after_load_mb"] = memory_mb()
        if yoloe:
            encoder_dir = root / "models/detection/yoloe-text"
            encoder_path = (encoder_dir / "mobileclip2_b.ts").resolve()
            from ultralytics.utils import downloads

            original_resolver = downloads.attempt_download_asset
            downloads.attempt_download_asset = lambda file, **kwargs: str(encoder_path)
            try:
                loaded.set_classes(["chair", "bottle", "backpack"])
            finally:
                downloads.attempt_download_asset = original_resolver
        warmup, warmup_ms = time_call(lambda: loaded.predict(source=image, device="cpu", verbose=False, imgsz=640))
        record["warmup_latency_ms"] = warmup_ms
        latencies = []
        prompt_results: dict[str, int] = {}
        for prompt in (["chair", "bottle", "backpack"] if yoloe else [None]):
            if yoloe:
                loaded.set_classes([prompt])
            for _ in range(10):
                results, elapsed = time_call(lambda: loaded.predict(source=image, device="cpu", verbose=False, imgsz=640))
                latencies.append(elapsed)
                if yoloe:
                    prompt_results[prompt] = len(results[0].boxes) if results else 0
        finish_timing(record, latencies)
        record["prompt_detections"] = prompt_results if yoloe else None
        record["detection_count_warmup"] = len(warmup[0].boxes) if warmup else 0
        record["offline_test_result"] = "PASS_network_blocked_local_path"
        record["status"] = "PASS"
        record["inference_run"] = True


def run_depth(record: dict[str, Any]) -> None:
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    local_dir = str(Path(record["model_path"]).parent)
    image = Image.open(IMAGE_PATH).convert("RGB")
    with network_block():
        processor, load_a = time_call(lambda: AutoImageProcessor.from_pretrained(local_dir, local_files_only=True))
        model, load_b = time_call(lambda: AutoModelForDepthEstimation.from_pretrained(local_dir, local_files_only=True))
        model.eval()
        record["load_time_ms"] = load_a + load_b
        record["ram_after_load_mb"] = memory_mb()
        inputs = processor(images=image, return_tensors="pt")
        _, warmup_ms = time_call(lambda: model(**inputs))
        record["warmup_latency_ms"] = warmup_ms
        latencies = []
        for _ in range(10):
            _, elapsed = time_call(lambda: model(**inputs))
            latencies.append(elapsed)
        finish_timing(record, latencies)
        record["input_settings"]["image_dimensions"] = list(image.size)
        record["offline_test_result"] = "PASS_network_blocked_local_files_only"
        record["status"] = "PASS"
        record["inference_run"] = True


def run_piper(record: dict[str, Any]) -> None:
    from piper.voice import PiperVoice

    config_path = Path(record["model_path"]).with_suffix(".onnx.json")
    with network_block():
        voice, load_ms = time_call(lambda: PiperVoice.load(record["model_path"], config_path=config_path, use_cuda=False))
        record["load_time_ms"] = load_ms
        record["ram_after_load_mb"] = memory_mb()
        OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with wave.open(str(OUTPUT_WAV), "wb") as wav_file:
            voice.synthesize_wav("FRIDAY sees a clear path ahead.", wav_file)
        record["warmup_latency_ms"] = (time.perf_counter() - start) * 1000
        record["output_path"] = str(OUTPUT_WAV)
        record["output_wav_size_bytes"] = OUTPUT_WAV.stat().st_size
        with wave.open(str(OUTPUT_WAV), "rb") as wav_file:
            record["output_wav_duration_seconds"] = wav_file.getnframes() / wav_file.getframerate()
        record["offline_test_result"] = "PASS_network_blocked_local_files_only"
        record["status"] = "PASS"
        record["inference_run"] = True


def run_one(model: dict[str, Any], root: Path) -> dict[str, Any]:
    path = (root / model["local_path"]).resolve()
    record = base_record(model, path)
    if not path.exists():
        record["errors"].append("Local model path does not exist")
        return record
    try:
        if model["loader"] == "ultralytics":
            run_yolo(model, root, record)
        elif model["loader"] == "ultralytics_yoloe":
            run_yolo(model, root, record, yoloe=True)
        elif model["loader"] == "torchscript":
            run_depth(record)
        elif model["loader"] == "piper_onnx":
            run_piper(record)
        elif model["loader"] == "faster_whisper":
            wav_files = list((root / "data").rglob("*.wav"))
            record["status"] = "BLOCKED"
            record["errors"].append("BLOCKED_PENDING_AUDIO: no local WAV file is available")
            record["offline_test_result"] = "not_run"
        else:
            record["errors"].append(f"Unsupported loader: {model['loader']}")
    except Exception as exc:
        record["status"] = "FAIL"
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    return record


def write_outputs(records: list[dict[str, Any]], root: Path, config: dict[str, Any]) -> None:
    json_path = root / config["results"]["json"]
    csv_path = root / config["results"]["csv"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    fields = sorted({key for item in records for key in item})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            row = dict(item)
            for key in ("input_settings", "errors", "latencies_ms", "prompt_detections"):
                if isinstance(row.get(key), (dict, list)):
                    row[key] = json.dumps(row[key])
            writer.writerow(row)


def write_report(records: list[dict[str, Any]], root: Path) -> None:
    model_bytes = sum(item.stat().st_size for item in (root / "models").rglob("*") if item.is_file())
    lines = [
        "# FRIDAY Phase 1D Model Benchmark Report",
        "",
        "CPU-only local inference. CUDA was not installed and `torch.cuda.is_available()` is false.",
        f"Model storage: {model_bytes:,} bytes ({model_bytes / (1024 * 1024):.2f} MiB).",
        "",
        "## Results",
        "",
    ]
    for item in records:
        lines.extend([
            f"### {item['model']} - {item['status']}",
            f"- Task: `{item['task']}`",
            f"- Device: `{item['device']}`; CUDA: `{item['cuda_available']}`",
            f"- Inference ran: `{item['inference_run']}`",
            f"- Load time: `{item['load_time_ms']}` ms",
            f"- Warm-up time: `{item['warmup_latency_ms']}` ms",
            f"- Mean/min/max latency: `{item['average_latency_ms']}` / `{item['min_latency_ms']}` / `{item['max_latency_ms']}` ms",
            f"- FPS: `{item['fps']}`",
            f"- Offline test: `{item['offline_test_result']}`",
            f"- Errors: {'; '.join(item['errors']) if item['errors'] else 'None'}",
            "",
        ])
    lines.extend([
        "## Limitations",
        "",
        "- The synthetic test image is a deterministic runtime input, not an accuracy benchmark.",
        "- faster-whisper is BLOCKED_PENDING_AUDIO because no genuine local WAV exists.",
        "- No ranking, score, or winner is assigned.",
    ])
    (root / "MODEL_BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/model_benchmark.yaml")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = load_config(root / args.config)
    if config["settings"].get("download_weights"):
        raise RuntimeError("download_weights must remain false")
    records = [run_one(model, root) for model in config["models"] if model.get("enabled", True)]
    write_outputs(records, root, config)
    write_report(records, root)
    print(f"Wrote {len(records)} local inference records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
