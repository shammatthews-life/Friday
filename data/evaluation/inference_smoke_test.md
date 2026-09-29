# FRIDAY Phase 1D Inference Smoke Test

## Environment

- Python: 3.14.5
- Device: CPU
- PyTorch: 2.14.0+cpu
- `torch.cuda.is_available()`: `False`
- Test image: `data/test_images/scene_common_objects.png`
- Network: socket connections blocked during model load/inference

## Results

- YOLO26n: PASS. Local detection inference completed; warm-up and 10 repeated CPU iterations recorded in `benchmark_results.json`.
- YOLOE-26n-seg: FAIL. The supplied `models/detection/yoloe-26n-seg.pt` is a TorchScript archive, not an Ultralytics PyTorch YOLOE checkpoint. Text prompts `chair`, `bottle`, and `backpack` were not executed. No replacement model was downloaded.
- Depth Anything V2 Small: PASS. Local Transformers depth inference completed; warm-up and 10 repeated CPU iterations recorded.
- faster-whisper tiny: BLOCKED_PENDING_AUDIO. No genuine local WAV exists, so transcription was not attempted.
- Piper en_US-amy-medium: PASS. Local synthesis completed.

## Piper output

- WAV: `data/evaluation/piper_amy_smoke_test.wav`
- Fixed sentence: `FRIDAY sees a clear path ahead.`
- Output duration: 2.461315 seconds
- Output size: 108,588 bytes

No unintended model files appeared. Inference did not use CUDA or a network URL.