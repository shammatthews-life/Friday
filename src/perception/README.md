# VisionAid Perception Pipeline

This package accepts timestamped camera, image, or video frames and returns a fused structured `SceneSnapshot` while preserving the existing `SceneState` used by FRIDAY, SceneMemory, and ObjectSearch.

## Architecture

```text
PerceptionFrame (image + timestamp + source)
  -> lazy local YOLO detector
  -> one-to-one IoU/center tracker
  -> scheduled relative depth (optional)
  -> conservative spatial relations + SceneMemory update
  -> SceneSnapshot / existing SceneState
  -> VisionAidBridge
```

`PerceptionPipeline` runs these stages sequentially. It does not own FRIDAY dialogue or make safety decisions. SceneObject retains its original constructor/API and has optional box, track, vertical-position, and relative-depth metadata. The existing `SafetyEngine` remains authoritative and independent from Qwen; unknown evidence must remain UNKNOWN. Guidance remains deterministic alignment logic, not route planning or certified collision avoidance.

`PerceptionFrame` is source-neutral: callers provide an ndarray plus timestamp/source identity, so the same pipeline accepts a webcam frame, image, or sampled video frame. `CameraSource` is an optional DirectShow source with configurable resolution, bounded warm-up, dark/blank/invalid rejection, timestamps, and explicit cleanup.

The existing `SceneMemory` is still updated for recent visibility and disappearance queries. The new tracker handles per-frame box association and re-identification; tracker IDs are distinct from SceneMemory's own historical IDs. ObjectSearch prefers a stable track ID when both target and detections have one, and retains its box/position fallback for existing callers.

## Profiles

Edit `configs/perception.yaml`; model paths are configuration values, not source-code constants. Create an unloaded pipeline with `create_pipeline("balanced")` and supply `PerceptionFrame` objects to `process()`.

| Profile | Detector | Input size | Depth schedule | Intended use |
|---|---|---:|---:|---|
| `quality` | local YOLO26n | 640 | every 2 frames | More frequent relative-depth refresh |
| `balanced` | local YOLO26n | 640 | every 5 frames | Default compromise |
| `realtime` | local YOLO26n | 480 | every 15 frames | Lower compute and depth frequency |

YOLO26n is the only compatible, locally staged closed-set detector measured here, so all profiles use it. `quality` currently means more frequent depth analysis, not a stronger or more accurate detector. YOLO26s is not staged or tested and was not downloaded. The profiles specify CPU because the installed PyTorch is `2.14.0+cpu` and `torch.cuda.is_available()` is false; the NVIDIA Vulkan path belongs to llama.cpp, not this Ultralytics/PyTorch detector.

Detector and depth model objects are lazy. Depth Anything V2 Small is created only on a scheduled depth request and, by default, released after that request. The detector remains resident while a pipeline session is active. The Qwen integration used about 5.01 GB peak process-tree RAM and left about 374 MB system RAM available; do not run Qwen concurrently with YOLO/depth on this laptop without an explicit memory plan. Close the pipeline when finished. YOLO inference and depth inference are sequential, never parallel.

## Depth, tracking, and relations

Depth Anything V2 Small is relative depth only. Scene objects may be labeled `relatively near`, `relatively middle-distance`, or `relatively far`; those labels are not metres/feet and are not calibrated. Failed inference, missing boxes, invalid crops, missing results, or invalid frames degrade to unknown/unavailable depth while retaining valid detections where possible.

The tracker associates same-label objects one-to-one using IoU and normalized center distance. It separates spatially distinguishable duplicate labels and records appeared, remained, moved, disappeared, and reacquired lifecycle information. Movement is thresholded to suppress small camera jitter; it is heuristic tracking, not a learned identity/re-identification system. Old tracks expire after a bounded re-identification window.

Spatial relations are emitted only for separated bounding boxes (`left_of`, `above`) or distinct known relative-depth categories (`closer_than`). Overlapping or insufficiently separated boxes do not create a relation. No physical distance is inferred.

## Models and YOLOE status

- Closed-set detector: local `models/detection/yolo26n.pt`.
- Relative depth: `models/depth/depth-anything-v2-small/`.
- Installed Ultralytics: 8.4.157. Its `YOLO` factory dispatches YOLOE-named `.pt`/YAML model definitions to `YOLOE`; the YOLOE API documents native Ultralytics `.pt` checkpoints or YAML definitions.
- The existing `models/detection/yoloe-26n-seg.pt` is a TorchScript ZIP archive (968 entries) and was rejected by the installed YOLOE loader. It was not overwritten, deleted, converted, or downloaded again. A TorchScript archive is not safely convertible to the native YOLOE training checkpoint from the available metadata. If revisited, stage a separately named official native `yoloe-26n-seg.pt` checkpoint at `models/detection/` only after approval; verify format and checksum first. YOLOE remains optional and is not used by normal closed-set detection.
- Text/visual-prompt adapters can implement the separate `PromptedDetector` protocol; no compatible YOLOE adapter is enabled by default.

## Tests and benchmarks

Deterministic tests use synthetic frames and mocked detector/depth providers:

```powershell
D:\friday\.venv\Scripts\python.exe scripts/test_perception_pipeline.py
```

Run the controlled local-image benchmark (no camera required):

```powershell
D:\friday\.venv\Scripts\python.exe scripts/benchmark_vision_pipeline.py --profile balanced --iterations 10
```

Add a bounded real-camera measurement with the known DirectShow backend:

```powershell
D:\friday\.venv\Scripts\python.exe scripts/benchmark_vision_pipeline.py --profile balanced --iterations 10 --camera-seconds 5
```

The benchmark reports model load separately from warm detector latency, depth load/latency, detection counts/confidence, tracker/fusion/full-pipeline overhead, camera FPS when requested, process RSS, and available system-wide NVIDIA memory samples. It writes ignored JSON detail and a human-readable Markdown report under `data/evaluation/`. These are runtime measurements, not accuracy benchmarks. Only the model variants actually present and run are reported.

## Measured v0.3 smoke benchmark

Run: `scripts/benchmark_vision_pipeline.py --profile balanced --iterations 10 --camera-seconds 5`. Input image was the checked-in controlled fixture `data/test_images/scene_common_objects.png`; camera frames were processed in memory and not saved.

- Profile/model: balanced / YOLO26n, CPU, 640px; YOLO26n load 87.84 ms.
- First detector/pipeline call: 5,925.91 ms (cold inference/runtime initialization; not representative of warm detector latency).
- Warm detector: mean 90.34 ms, 11.07 FPS; controlled-image mean detection count 1.0 and confidence 0.581.
- Depth Anything V2 Small: reported model load 292.87 ms; first scheduled depth call 5,946.21 ms; two controlled-image calls averaged 3,958.09 ms. In the camera segment, two scheduled calls averaged 1,963.82 ms. First-call timing includes lazy runtime initialization beyond the estimator's reported weight-load timer.
- Tracking: 0.087 ms mean. Fusion: 0.176 ms mean.
- Complete controlled-image pipeline: 896.44 ms mean / 1.12 FPS across 10 iterations; range 77.66–6,026.69 ms.
- CAP_DSHOW at actual 640x480: PASS; raw capture measured 150 frames in 5.01 s (29.97 FPS). The pipeline processed 10 frames in 5.17 s (1.99 FPS); a person was detected at mean confidence 0.915 (best 0.921).
- Peak process RSS: 791,937,024 bytes; minimum available system RAM 1,971,838,976 bytes during this run.
- `nvidia-smi` system-wide sample: 0 MiB used before/after; PyTorch CUDA unavailable, so detector/depth ran on CPU.

The detector itself is near 11 warm FPS, but scheduled relative depth limits the balanced fused camera pipeline to about 2 FPS. **Recommended profile for this checkpoint: `balanced`**, because it produced real-camera detections in the measured run. The realtime profile was also sampled: its 480px controlled-image run produced zero detections on that fixture; raw camera capture was about 29.95 FPS, while only 4 pipeline frames were processed in 5.02 s (0.81 FPS) because the first scheduled depth call took 4.72 s. Its steady-state advantage remains unverified. The quality profile was not benchmarked. Historical Phase 1D results used a different harness and are not mixed into these measurements.

## Limitations

- No stronger compatible YOLO26 checkpoint is local, so detector quality profiles cannot yet compare model variants.
- Camera detection and relative depth quality have not been calibrated across lighting, motion, or varied real-world scenes.
- Relative-depth categories are heuristic image-relative ordering, not physical distance.
- IoU/center tracking can switch identities when same-class objects overlap or cross.
- Camera blank-frame thresholds and movement thresholds are configurable heuristics and need broader evaluation.
- The safety engine is prototype logic, not certified collision avoidance. Qwen does not calculate or override safety state.
