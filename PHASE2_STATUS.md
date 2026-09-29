# FRIDAY / VISIONAID — Phase 2 Status Checkpoint

**Checkpoint date:** 2026-09-29
**Scope:** Documentation-only status record. No hardware, camera, microphone, inference, or model checks were run for this checkpoint.

## 1. Completed Phases

| Phase | Scope | Status |
|---|---|---|
| Phase 0 | Environment audit | PASS |
| Phase 0.5 | Hardware / PyTorch | PASS; CUDA unavailable |
| Phase 1A | Model registry | PASS |
| Phase 1B | Model download / integrity | PASS |
| Phase 1C | Benchmark framework | PASS |
| Phase 1D | Local model inference | PASS / PARTIAL: YOLO26n, relative depth, and Piper inference passed; YOLOE remains incompatible/unverified and Whisper is blocked |
| Phase 2A | FRIDAY core | PASS |
| Phase 2B | On-demand relative depth | PASS |
| Phase 2C | Real webcam + YOLO | PASS |
| Phase 2D.1 | Microphone capture | PASS |
| Phase 2D.2 | faster-whisper | BLOCKED by Windows Code Integrity |
| Phase 2D.3 | Piper TTS | PASS |
| Phase 2D.4 | Live typed FRIDAY + Piper | PASS |
| Phase 2E | ObjectSearch / TargetLock | PASS |
| Phase 2F | SceneMemory | PASS |
| Phase 2G | FRIDAY + SceneMemory | PASS |
| Phase 2H | Conversational context | PASS |
| Phase 2I | Target guidance logic | PASS |
| Phase 2J | Independent safety engine | PASS; prototype logic only |
| Phase 2K | Deterministic core integration | PASS |
| Phase 2L | Automated live end-to-end smoke test | PASS with limitations |

## 2. Verified Live Pipeline

**Real webcam → YOLO26n → SceneState → FRIDAY → ObjectSearch / relative depth → Piper TTS**

The live smoke test exercised local camera frames through YOLO and FRIDAY. Object search and relative depth remain existing on-demand integrations; Piper synthesized the FRIDAY responses locally.

## 3. Live Test Evidence

Latest automated test evidence:

- Camera backend: `CAP_DSHOW`
- Resolution: `640x480`
- Real detections included a person, a clock, and bottles
- Four FRIDAY query attempts completed in the requested order
- Four local Piper WAV files were generated under `data/test_audio/live_automated/`
- Annotated frame generated at `data/test_images/friday_live_automated.jpg`
- The exact Q3 terminal response was not captured in the latest run output

## 4. Important Limitations

- PyTorch is CPU-only: `2.14.0+cpu`.
- CUDA is unavailable.
- faster-whisper is blocked by Windows Code Integrity.
- The supplied YOLOE checkpoint remains incompatible / unverified.
- Depth Anything V2 Small provides relative depth, not calibrated metres or feet.
- Synthetic images are not accuracy benchmarks.
- The safety engine is prototype logic, not certified collision avoidance.
- The exact automated Q3 terminal response was not captured in the latest run.

## 5. Current Windows Runtime State

- Smart App Control was turned OFF by the user through Windows Security.
- PyTorch import subsequently works.
- PyTorch version: `2.14.0+cpu`.
- CUDA remains unavailable.
- This checkpoint makes no security-policy change or bypass recommendation. The preceding state recorded in older status notes is superseded by this user-confirmed update.

## 6. Minor Known Bug

FRIDAY pluralization can produce `2 bottle` instead of `2 bottles`. Record this as a minor future cleanup item; it is not part of this checkpoint's scope.

## 7. Next Major Work

Choose the next phase deliberately from these options:

- Resolve local Whisper / STT
- Obtain GPU / CUDA acceleration
- Correct or replace the YOLOE workflow
- Conduct deeper real-world evaluation
- Plan eventual Android deployment

No next-phase work is implemented by this checkpoint.
