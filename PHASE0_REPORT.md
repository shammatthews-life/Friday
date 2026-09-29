# PHASE 0 REPORT

## Verified facts

- Python version: 3.14.5
- Python architecture: 64-bit AMD64
- Platform: Windows / win32
- The local machine audit script was created and the project scaffold was generated.
- Full CPU/GPU/RAM/storage/audio/webcam detection could not be conclusively captured in this shell session because the Windows terminal returned incomplete or non-echoed output for WMI and shell commands.

## A. Hardware detected

- Operating system: Windows (confirmed as Windows / win32 from the Python platform data)
- Exact Windows release, CPU model, RAM, GPU model, VRAM, storage, webcam, microphone, and speaker inventory: not conclusively verified in this environment because the native shell did not return the required WMI output reliably.

## B. Python environment

- Python: 3.14.5
- Architecture: 64-bit
- Platform: win32

## C. GPU/CUDA status

- CUDA status is not verified here.
- No reliable GPU model or VRAM value was captured in this session.
- A local CUDA check should be run in the normal Windows shell before any large model downloads.

## D. Model candidates

- Object detection: YOLO11n, YOLO11s, YOLO11m
- Open-vocabulary search: YOLOE-n, YOLOE-s, YOLOE-m
- Depth: Depth Anything V2 Small / Base / Large
- STT: faster-whisper tiny / small / medium
- TTS: Piper ONNX local voice set
- Optional local LLM: small Gemma/Qwen/Phi only if hardware permits

## E. Approximate storage requirements

- YOLO11n: ~20-40 MB
- YOLO11s: ~40-80 MB
- YOLOE-n / YOLOE-s: moderate, typically tens to low hundreds of MB
- Depth Anything V2 Small: ~100-300 MB
- faster-whisper small: ~500 MB to 1.5 GB depending on weights
- Piper voice: ~50-200 MB per voice
- Initial benchmark set total: roughly 1-3 GB, without large optional LLMs

## F. Which models should be downloaded first

1. YOLO11n
2. YOLO11s
3. YOLOE-n or YOLOE-s
4. Depth Anything V2 Small
5. faster-whisper small
6. a small Piper ONNX voice

This is the smallest representative benchmark set for phase 1.

## G. What Phase 1 will implement

- Benchmark object detection locally
- Benchmark open-vocabulary search mode
- Benchmark depth models for indoor and outdoor scenes
- Benchmark STT models on the required commands
- Benchmark TTS latency and quality
- Build the local benchmark pipeline and measurement report
- Select the best small models before any broader download sweep

## Project scaffold created

- [README.md](README.md)
- [MODEL_REGISTRY.md](MODEL_REGISTRY.md)
- [MODEL_BENCHMARKS.md](MODEL_BENCHMARKS.md)
- [configs/device.yaml](configs/device.yaml)
- [requirements.txt](requirements.txt)
- [pyproject.toml](pyproject.toml)
- [src/main.py](src/main.py)
