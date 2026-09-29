# FRIDAY Phase 1D Model Benchmark Report

CPU-only local inference. CUDA was not installed and `torch.cuda.is_available()` is false.
Model storage: 511,634,833 bytes (487.93 MiB).

Runtime packages installed and verified for Python 3.14.5: `ultralytics==8.4.157`, `transformers==5.17.0`, `faster-whisper==1.2.1`, `piper-tts==1.8.0`, `onnxruntime==1.30.0`, `psutil==7.2.2`, `soundfile==0.14.0`, and the pinned Ultralytics CLIP commit. Existing `torch==2.14.0+cpu` and `torchvision==0.29.0+cpu` were preserved.

## Results

### YOLO26n - PASS
- Task: `object_detection`
- Device: `cpu`; CUDA: `False`
- Inference ran: `True`
- Load time: `75.44770000095014` ms
- Warm-up time: `1912.7751000014541` ms
- Mean/min/max latency: `59.386159999121446` / `54.40529999759747` / `83.73840000058408` ms
- FPS: `16.838940251647756`
- Offline test: `PASS_network_blocked_local_path`
- Errors: None

### YOLOE-26n-seg - FAIL
- Task: `open_vocabulary_search`
- Device: `cpu`; CUDA: `False`
- Inference ran: `False`
- Load time: `None` ms
- Warm-up time: `None` ms
- Mean/min/max latency: `None` / `None` / `None` ms
- FPS: `None`
- Offline test: `not_run`
- Errors: The supplied local `yoloe-26n-seg.pt` is a TorchScript archive, not an Ultralytics PyTorch checkpoint. Ultralytics rejected it before prompt inference; no replacement model was downloaded.
Load the original .pt weights, or export again with format='torchscript' and load that file directly.

### Depth Anything V2 Small - PASS
- Task: `depth`
- Device: `cpu`; CUDA: `False`
- Inference ran: `True`
- Load time: `401.27740000389167` ms
- Warm-up time: `1251.160300002084` ms
- Mean/min/max latency: `1173.3877099999518` / `1122.9705000005197` / `1268.1163000015658` ms
- FPS: `0.8522332315889358`
- Offline test: `PASS_network_blocked_local_files_only`
- Errors: None

### faster-whisper tiny - BLOCKED
- Task: `speech_to_text`
- Device: `cpu`; CUDA: `False`
- Inference ran: `False`
- Load time: `None` ms
- Warm-up time: `None` ms
- Mean/min/max latency: `None` / `None` / `None` ms
- FPS: `None`
- Offline test: `not_run`
- Errors: BLOCKED_PENDING_AUDIO: no local WAV file is available

### en_US-amy-medium - PASS
- Task: `text_to_speech`
- Device: `cpu`; CUDA: `False`
- Inference ran: `True`
- Load time: `3184.8307000000204` ms
- Warm-up time: `916.4601000011317` ms
- Mean/min/max latency: `None` / `None` / `None` ms
- FPS: `None`
- Offline test: `PASS_network_blocked_local_files_only`
- Errors: None

## Limitations

- The synthetic test image is a deterministic runtime input, not an accuracy benchmark.
- faster-whisper is BLOCKED_PENDING_AUDIO because no genuine local WAV exists.
- YOLOE text-prompt inference is blocked by the downloaded checkpoint format; the local `mobileclip2_b.ts` encoder is present and the resolver was forced local, but no prompt inference ran.
- No ranking, score, or winner is assigned.
