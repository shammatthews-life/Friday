# FRIDAY Phase 1A Benchmark Specification

This document defines measurements only. No benchmark is run in Phase 1A and no model weights are downloaded. The current environment has CPU-only PyTorch (`CUDA VERIFIED = NO`), so a later run must record CPU explicitly and must never assume CUDA.

## Candidate groups

- Continuous detection: YOLO26n, YOLO26s
- Search detection: YOLOE-26n-seg, YOLOE-26s-seg
- Depth: Depth Anything V2 Small, Depth Anything V2 Base
- Speech-to-text: faster-whisper tiny, faster-whisper small
- Text-to-speech: en_US-lessac-medium, en_US-amy-medium
- Optional LLM: Qwen3-1.7B, Phi-4-mini-instruct

Depth Anything V2 Large is excluded from this phase because the machine has approximately 6 GB VRAM. Optional LLMs are not part of the perception or safety path.

## Required result fields

Every result record must contain these fields, using `null` or `Not measured` rather than invented values:

### Identity and setup

- `timestamp_utc`
- `model_name`
- `family`
- `capability`
- `mode` (`continuous`, `search`, or `on_demand`)
- `model_path`
- `model_file_size_bytes` and `model_file_size_mb`
- `status` (`passed`, `missing_weight`, `unsupported`, or `failed`)
- `error`
- `device_runtime` (`cpu`, `cuda`, or another explicitly named runtime)
- `cuda_available`
- `device_name`
- `batch_size`
- `image_resolution` where applicable
- `audio_sample_rate_hz`, `audio_channels`, and `audio_duration_seconds` where applicable

### Timing and resource measurements

- `model_load_time_ms`
- `warmup_latency_ms`
- `repeated_run_count`
- `repeated_run_latency_ms`
- `average_latency_ms`
- `p50_latency_ms`
- `p95_latency_ms`
- `fps` where applicable
- `ram_before_mb`, `ram_after_load_mb`, and `ram_peak_mb` where measurable
- `vram_before_mb`, `vram_after_load_mb`, and `vram_peak_mb` where measurable
- `cpu_usage_percent` where measurable

### Quality measurements

- Detection/search: precision, recall, mAP or task-specific detection score on a fixed labeled set; prompt-hit rate for open-vocabulary search.
- Depth: official-compatible depth metric where ground truth exists, plus ordinal depth accuracy and failure count for safety-relevant scenes.
- Speech-to-text: word error rate, command exact-match rate, false activation rate, and transcription latency.
- Text-to-speech: real-time factor, synthesis latency, intelligibility/user rating, and audio underrun count.
- LLM: command-routing exact match, tool-call schema validity, grounded response rate, and unsafe suggestion count. It is never a safety authority.

## Measurement rules

1. Record model file size separately from runtime RAM and runtime VRAM. A weight file on disk is not a memory measurement.
2. Use one model process at a time and close it before loading the next candidate.
3. Record cold load time once, then run at least one warm-up inference that is excluded from repeated averages.
4. Run a configured number of repeated inferences with fixed inputs and settings; report all samples plus average, p50, and p95.
5. Use batch size 1 for FRIDAY real-time comparisons unless a candidate explicitly requires another setting.
6. Record the exact image resolution, audio format, sample rate, channels, beam size, compute type, quantization, and thread count.
7. Synchronize CUDA before and after timed work when CUDA is available. On CPU, use `perf_counter` without CUDA calls.
8. If CUDA is unavailable, benchmark CPU only and preserve `cuda_available: false`; do not silently fall back while labeling a result CUDA.
9. Missing local weights must produce a structured `missing_weight` record. The runner must not download by model name or URL.
10. Quality scores require a fixed, versioned evaluation set. Latency without quality is not a model selection decision.

## Recommended phase order

1. Verify the environment gate and record CPU/CUDA state.
2. Benchmark YOLO26n and YOLO26s at the same image resolution.
3. Benchmark YOLOE-26n-seg and YOLOE-26s-seg with the same prompt list and search images.
4. Benchmark Depth Anything V2 Small, then Base if Small does not meet the quality target.
5. Benchmark faster-whisper tiny and small with identical command audio.
6. Benchmark the two Piper voices for latency and intelligibility.
7. Consider the optional LLMs only after perception and safety paths are stable.
