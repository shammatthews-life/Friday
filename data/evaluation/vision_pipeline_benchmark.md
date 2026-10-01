# VisionAid Perception Pipeline Benchmark

Controlled local image benchmark; smoke-test measurements, not an accuracy benchmark.

- Status: **PASS**
- Profile: `balanced`
- Detector: `YOLO26n` (`models/detection/yolo26n.pt`)
- Device: `cpu`
- Model variants actually tested: YOLO26n
- Stronger local detector checkpoint: YOLO26s not locally staged; not downloaded or tested

## Controlled Image Measurements

- Input: `data\test_images\scene_common_objects.png`
- Model load: 87.84019999438897 ms
- Initial frame (cold pipeline): 5925.914499995997 ms; scheduled depth: 0.0 ms
- First scheduled depth warm-up: {'status': 'measured_in_controlled_iterations'}
- Detector latency: 90.338080006768 ms mean; 11.069529039415952 FPS
- Detection count: mean 1.0; confidences: {'count': 10, 'mean': 0.580725908279419, 'minimum': 0.580725908279419, 'maximum': 0.580725908279419}
- Depth load: 292.86640000646 ms
- Depth latency: 3958.0885999894235 ms mean
- First scheduled depth call: 5946.211699978448 ms
- Tracking overhead: 0.08687000372447073 ms mean
- Fusion overhead: 0.17647000204306096 ms mean
- Complete pipeline: 896.4362800092204 ms mean; 1.1155282559399697 FPS

## Camera Measurements

- Status: PASS
- Backend/resolution: CAP_DSHOW / requested [640, 480], actual [640, 480]
- Raw capture: 150 frames in 5.005419199995231 s; 29.967520003148373 FPS
- Pipeline: 10 frames in 5.1705166999890935 s; 1.9873990553131327 FPS
- Depth calls during camera pipeline: 2; first 1975.375299982261 ms; mean 1963.817099982407 ms
- Camera detections: mean count 1.0; best confidences {'person': 0.9207663536071777}

## Memory

- Peak process RSS: 791937024 bytes
- Minimum available system RAM: 1971838976 bytes
- GPU memory samples: {"before": {"name": "NVIDIA GeForce RTX 4050 Laptop GPU", "total_mib": 6141, "used_mib": 0, "free_mib": 5920}, "after": {"name": "NVIDIA GeForce RTX 4050 Laptop GPU", "total_mib": 6141, "used_mib": 0, "free_mib": 5920}}

## Model Compatibility and Limits

- YOLO26n is the only locally staged compatible closed-set detector tested by this benchmark.
- YOLO26s is not locally staged; it was not downloaded or tested.
- YOLOE-26n-seg remains an incompatible TorchScript archive for the installed Ultralytics loader; it was not replaced or modified.
- Depth Anything V2 Small reports scene-relative depth only, never physical distance.
- CUDA is unavailable to PyTorch in this environment; YOLO and depth measurements use CPU.
- Synthetic/local image results do not establish real-world accuracy or certified safety.
