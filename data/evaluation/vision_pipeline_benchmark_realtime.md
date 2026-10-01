# VisionAid Perception Pipeline Benchmark

Controlled local image benchmark; smoke-test measurements, not an accuracy benchmark.

- Status: **PASS**
- Profile: `realtime`
- Detector: `YOLO26n` (`models/detection/yolo26n.pt`)
- Device: `cpu`
- Model variants actually tested: YOLO26n
- Stronger local detector checkpoint: YOLO26s not locally staged; not downloaded or tested

## Controlled Image Measurements

- Input: `data\test_images\scene_common_objects.png`
- Model load: 87.38360001007095 ms
- Initial frame (cold pipeline): 5852.8882000246085 ms; scheduled depth: 0.0 ms
- First scheduled depth warm-up: {'status': 'whole_frame_depth_warmup_only', 'inference_ms': 5767.44370002416, 'not_a_detection': True}
- Detector latency: 56.90517000039108 ms mean; 17.5730957308998 FPS
- Detection count: mean 0.0; confidences: {'count': 0, 'mean': None, 'minimum': None, 'maximum': None}
- Depth load: 280.4525999817997 ms
- Depth latency: None ms mean
- First scheduled depth call: None ms
- Tracking overhead: 0.021520000882446766 ms mean
- Fusion overhead: 0.04368999507278204 ms mean
- Complete pipeline: 70.35686000017449 ms mean; 14.213255111122354 FPS

## Camera Measurements

- Status: PASS
- Backend/resolution: CAP_DSHOW / requested [640, 480], actual [640, 480]
- Raw capture: 150 frames in 5.006796500005294 s; 29.95927635561809 FPS
- Pipeline: 19 frames in 5.8376691000012215 s; 3.3973045785973106 FPS
- Depth calls during camera pipeline: 2; first 1813.1157000025269 ms; mean 1890.7637499942211 ms
- Camera detections: mean count 1.0; best confidences {'person': 0.9296990036964417}

## Memory

- Peak process RSS: 738070528 bytes
- Minimum available system RAM: 1890410496 bytes
- GPU memory samples: {"before": {"name": "NVIDIA GeForce RTX 4050 Laptop GPU", "total_mib": 6141, "used_mib": 0, "free_mib": 5920}, "after": {"name": "NVIDIA GeForce RTX 4050 Laptop GPU", "total_mib": 6141, "used_mib": 0, "free_mib": 5920}}

## Model Compatibility and Limits

- YOLO26n is the only locally staged compatible closed-set detector tested by this benchmark.
- YOLO26s is not locally staged; it was not downloaded or tested.
- YOLOE-26n-seg remains an incompatible TorchScript archive for the installed Ultralytics loader; it was not replaced or modified.
- Depth Anything V2 Small reports scene-relative depth only, never physical distance.
- CUDA is unavailable to PyTorch in this environment; YOLO and depth measurements use CPU.
- Synthetic/local image results do not establish real-world accuracy or certified safety.
