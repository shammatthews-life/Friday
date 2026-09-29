# HARDWARE AUDIT

## Overview

This audit covers the local desktop machine used for the VisionAid / FRIDAY prototype. The goal is to establish the real system profile before selecting local benchmark models. No model downloads or training were performed in this phase.

## 1. Operating System and Python

- OS: Microsoft Windows 11 Home Single Language
- Version: 10.0.26200
- Python: 3.14.5
- Architecture: 64-bit AMD64
- Python executable: C:\Users\SHARON\AppData\Local\Programs\Python\Python314\python.exe

## 2. CPU

- CPU name: AMD Ryzen 7 7445HS w/ Radeon 740M Graphics
- Physical cores: 6
- Logical processors: 12
- Max clock speed: 3201 MHz

## 3. RAM

- Total physical RAM: 15,357 MB (approx. 15.0 GB)
- Available RAM: UNKNOWN from the live system at this point; the Windows report shows free physical memory, but this was not recorded in a stable persistent artifact in the session.
- Current memory usage: UNKNOWN

## 4. GPU

- GPU vendor: NVIDIA
- GPU model: NVIDIA GeForce RTX 4050 Laptop GPU
- Dedicated VRAM: 4,096 MB (approx. 4.0 GB)
- Integrated GPU: AMD Radeon 740M Graphics
- Integrated/shared memory: 512 MB
- Driver version: 32.0.15.9282
- NVIDIA SMI status: NOT VERIFIED in this session

## 5. CUDA and PyTorch

- CUDA available: UNKNOWN
- PyTorch installation: UNKNOWN in the project venv from a validated import test
- torchvision: UNKNOWN in the project venv from a validated import test
- torch.cuda.is_available(): UNKNOWN
- torch.version.cuda: UNKNOWN
- visible CUDA devices: UNKNOWN
- GPU name reported by PyTorch: UNKNOWN

## 6. Storage

- C: total 296 GB, free 150 GB
- D: total 107 GB, free 100 GB
- E: total 107 GB, free 100 GB
- D:\friday filesystem has roughly 100 GB free in the current workstation environment

## 7. Camera

- Detected: Yes
- Name: HP Wide Vision HD Camera
- Camera index: UNKNOWN
- Resolution: UNKNOWN

## 8. Microphone

- Detected: Yes
- Name: Microphone Array (AMD Audio Device)
- Sample rate: UNKNOWN
- Channel count: UNKNOWN

## 9. Audio Output

- Detected: Yes
- Output device: Speaker (Realtek(R) Audio)

## 10. Preliminary capability profile

- COMPUTE_TIER: MEDIUM
- GPU_INFERENCE: YES (NVIDIA RTX 4050 Laptop GPU present)
- LOCAL_LLM_FEASIBLE: NEEDS_TESTING
- DEPTH_LARGE_FEASIBLE: NEEDS_TESTING

These are preliminary recommendations only; they are not final model decisions.

## 11. Compatibility status for Python 3.14

The current project is using Python 3.14.5. PyTorch and other ML packages were not yet confirmed to import successfully from the venv in a stable final verification step, so compatibility is not proven. This means the system remains in a cautious state until a real import test confirms the package stack.

## 12. Model selection input

Recommended first benchmark set pending validated import compatibility:

- YOLO family: YOLO11n / YOLO11s
- Open-vocabulary search: YOLOE-n / YOLOE-s
- Depth: Depth Anything V2 Small
- STT: faster-whisper small
- TTS: Piper ONNX small voice

No large model downloads were approved in this phase.
