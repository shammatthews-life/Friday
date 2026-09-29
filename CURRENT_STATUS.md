# VisionAid / FRIDAY — Current Project Status

**Checkpoint date:** 2026-09-29
**Project root:** `D:\friday`

## 1. Project Identity

- **VisionAid** is the overall assistive vision product.
- **FRIDAY** is the AI assistant within VisionAid.
- The project is offline-first and currently developed in Python. Flutter/Android remains paused.

## 2. Project Location

`D:\friday`

## 3. Python Environment

Use this project virtual-environment interpreter:

`D:\friday\.venv\Scripts\python.exe`

Do not use global/system Python for project commands.

## 4. Hardware

- Operating system: Windows 11
- GPU: NVIDIA RTX 4050 Laptop GPU
- VRAM: approximately 6 GB, per the current project hardware profile
- RAM: approximately 15.6 GB

Hardware documentation has reported different dedicated VRAM figures. Treat the approximate figure above as a planning profile, not a newly measured value.

## 5. Python and PyTorch Status

- Python: 3.14.5
- PyTorch: `2.14.0+cpu`
- CUDA: unavailable; `torch.cuda.is_available()` is false.
- Current status: importing PyTorch is blocked when Windows Application Control attempts to load `torch.dll`.
- Do not retry CUDA installation or alter the Python version as part of this checkpoint.

## 6. Windows Application Control Blocker

The confirmed Code Integrity diagnosis is:

- Policy name: `VerifiedAndReputableDesktop`
- Policy ID: `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`
- Relevant Code Integrity Operational events: `3033` and `3077`
- Blocked native file: `D:\friday\.venv\Lib\site-packages\torch\lib\torch.dll`
- The `torch.dll` file was reported unsigned; the event recorded requested signing level 2 and validated signing level 1.
- CTranslate2's `_ext.cp314-win_amd64.pyd` was previously blocked under the same policy.

Do not bypass or disable this policy, weaken Windows security, or modify Group Policy. Resolve any runtime restriction only through legitimate, authorized channels.

## 7. Completed Phases

| Phase | Scope | Status |
|---|---|---|
| Phase 0 | Environment audit | PASS |
| Phase 0.5 | Hardware/PyTorch audit | PASS; CUDA unavailable |
| Phase 1A | Model registry | PASS |
| Phase 1B | Model download/integrity | PASS |
| Phase 1C | Benchmark framework | PASS |
| Phase 1D | Local inference | PASS/PARTIAL as previously documented; current native-runtime policy blocks live PyTorch-dependent execution |
| Phase 2A | FRIDAY core | PASS |
| Phase 2B | On-demand relative depth | PASS |
| Phase 2C | Real webcam/YOLO | PASS historically; currently blocked by the `torch.dll` Application Control restriction |
| Phase 2D.1 | Microphone capture | PASS |
| Phase 2D.2 | Whisper | BLOCKED by Windows Code Integrity |
| Phase 2D.3 | Piper TTS | PASS |
| Phase 2D.4 | Live typed FRIDAY + Piper | PASS historically |
| Phase 2E.2 | ObjectSearch/TargetLock logic | PASS |
| Phase 2E.3 | FRIDAY + search integration | PASS |
| Phase 2F | SceneMemory | PASS |
| Phase 2G | FRIDAY + SceneMemory | PASS |
| Phase 2H | Conversational context | PASS |
| Phase 2I | Target guidance logic | PASS |
| Phase 2J | Independent safety engine | PASS; prototype logic only |
| Phase 2K | Deterministic core integration | PASS; software test only |

Earlier benchmark and environment reports document tests performed before the current Application Control diagnosis. They are historical results and do not establish that live inference works under the current runtime restriction.

## 8. Working Core Pipeline

The intended and historically exercised core pipeline is:

**Camera → YOLO26n → SceneState → SceneMemory → FRIDAY → ObjectSearch / Guidance → Piper**

Current qualifications:

- Live PyTorch-dependent camera/YOLO execution is presently blocked by Windows Application Control.
- Relative depth is computed on demand, not continuously.
- Depth Anything V2 output is qualitative/relative only. No metres or feet are claimed.
- The prototype SafetyEngine is independent of FRIDAY conversational reasoning.
- Guidance and safety logic tests use deterministic mock inputs; they do not establish real-world performance or safety.

## 9. Model Status

- **YOLO26n:** Local model previously worked and was verified. Current live inference is blocked because the native PyTorch runtime cannot load `torch.dll`.
- **YOLOE-26n-seg:** Blocked/incompatible checkpoint format. The local file was identified as a TorchScript archive rejected by the current Ultralytics loader. Do not replace or download another YOLOE model unless explicitly authorized.
- **Depth Anything V2 Small:** Local checkpoint is present and was previously verified working. Its output is relative depth only; current execution depends on the blocked PyTorch runtime.
- **faster-whisper tiny:** Local model files are present. CTranslate2 native runtime is blocked by Windows Code Integrity.
- **Piper Amy medium:** Local model is present and Piper synthesis was previously verified working.

## 10. Current Blockers

Only the following blockers are recorded:

- Windows Application Control blocks the native `torch.dll` load.
- The same policy blocks the CTranslate2 native module used by faster-whisper.
- The downloaded YOLOE checkpoint format is incompatible with the current Ultralytics loader.
- Whisper cannot execute until the Windows native-runtime restriction is legitimately resolved.

## 11. Next Major Task

> Resolve or otherwise legitimately work around the Windows native-runtime/Application Control blocker before further live ML integration.

Future work must not bypass or disable the security policy. Any resolution must be legitimate and authorized by the applicable system/security administrators.

## 12. Recommended Feature Order After the Blocker Is Resolved

1. Restore/verify the PyTorch runtime.
2. Verify live YOLO again.
3. Verify real object search and target lock.
4. Integrate independent safety perception.
5. Resolve/test local Whisper.
6. Connect microphone → Whisper → FRIDAY → Piper.
7. Verify the end-to-end live FRIDAY voice loop.
8. Benchmark the optimized runtime.
9. Prepare mobile deployment architecture.

## 13. Credit-Efficiency Rule

> Future coding tasks must remain small, single-phase tasks. Do not combine multiple new capabilities into one Copilot request.

## 14. Validation Scope

This checkpoint records the project status and prior validation results; it does not run or claim a new model, camera, microphone, depth, speech, or hardware validation. The documentation write is the only action performed for this checkpoint.
