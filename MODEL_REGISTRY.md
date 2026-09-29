# FRIDAY Model Registry

Phase 1A was the research and benchmark plan. Phase 1B downloaded only the five initial benchmark candidates listed below. No inference or benchmarking has been run. The benchmark runner accepts local files only.

## Hardware and runtime constraints

- OS: Windows 11
- Python: 3.14.5, 64-bit
- CPU: AMD Ryzen 7 7445HS
- RAM: approximately 15.6 GB
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, approximately 6 GB reported by `nvidia-smi`
- PyTorch: CPU-only; `CUDA VERIFIED = NO`
- Selection rule: prefer small candidates and keep safety decisions outside any optional language model.

## Candidate summary

| Candidate | Family / capability | Mode | Parameters | Approx. weight size | License | Official source |
|---|---|---|---:|---:|---|---|
| YOLO26n | Ultralytics YOLO26 / continuous object detection | Continuous | 2.4M official fused detection configuration | Not documented | AGPL-3.0 or Ultralytics Enterprise terms | [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/) |
| YOLO26s | Ultralytics YOLO26 / continuous object detection | Continuous | 9.5M official fused detection configuration | Not documented | AGPL-3.0 or Ultralytics Enterprise terms | [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/) |
| YOLOE-26n-seg | Ultralytics YOLOE-26 / open-vocabulary detection and segmentation | Search | 3.9M official detection configuration; released segmentation checkpoint count not documented | Not documented | AGPL-3.0 or Ultralytics Enterprise terms | [YOLOE docs](https://docs.ultralytics.com/models/yoloe/) |
| YOLOE-26s-seg | Ultralytics YOLOE-26 / open-vocabulary detection and segmentation | Search | 10.7M official detection configuration; released segmentation checkpoint count not documented | Not documented | AGPL-3.0 or Ultralytics Enterprise terms | [YOLOE docs](https://docs.ultralytics.com/models/yoloe/) |
| Depth Anything V2 Small | Depth Anything V2 / monocular relative depth | Continuous support | 24.8M | Not documented | Apache-2.0 | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| Depth Anything V2 Base | Depth Anything V2 / monocular relative depth | Continuous support | 97.5M | Not documented | CC-BY-NC-4.0 | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| faster-whisper small | CTranslate2 Whisper conversion / speech-to-text | On demand | Not documented | Not documented | MIT model repository; review original Whisper terms separately | [Model card](https://huggingface.co/Systran/faster-whisper-small) |
| faster-whisper tiny | CTranslate2 Whisper conversion / speech-to-text | On demand | Not documented | Not documented | MIT model repository; review original Whisper terms separately | [Model card](https://huggingface.co/Systran/faster-whisper-tiny) |
| en_US-lessac-medium | Piper VITS ONNX / text-to-speech | On demand | 15-20M quality-band range | Not documented | Voice-specific; review its `MODEL_CARD` | [Piper voices](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md) |
| en_US-amy-medium | Piper VITS ONNX / text-to-speech | On demand | 15-20M quality-band range | Not documented | Voice-specific; review its `MODEL_CARD` | [Piper voices](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md) |
| Qwen3-1.7B | Qwen3 / optional local LLM | On demand | 1.7B | 4.08 GB repository size | Apache-2.0 | [Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B) |
| Phi-4-mini-instruct | Phi-4 / optional local LLM | On demand | 3.8B | 7.69 GB repository size | MIT | [Phi model card](https://huggingface.co/microsoft/Phi-4-mini-instruct) |

The parameter figures for YOLO26 are from the official fused detection table. The released YOLOE files are segmentation checkpoints with additional components, so the detection-table figures must not be treated as the full checkpoint parameter count. Piper's parameter range is documented for the `medium` quality band, not for each individual voice.

## Phase 1B downloaded set

| Candidate | Local path | Actual size | SHA-256 | Source checksum |
|---|---|---:|---|---|
| YOLO26n | `models/detection/yolo26n.pt` | 5,544,453 bytes (5.29 MiB) | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | Not provided by official release page |
| YOLOE-26n-seg | `models/detection/yoloe-26n-seg.pt` | 11,710,443 bytes (11.17 MiB) | `1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b` | Not provided by official release page |
| Depth Anything V2 Small | `models/depth/depth-anything-v2-small/model.safetensors` | 99,173,660 bytes (94.58 MiB) | `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70` | Matched official Hugging Face LFS SHA-256 |
| faster-whisper tiny | `models/speech/faster-whisper-tiny/model.bin` | 75,538,270 bytes (72.04 MiB) | `dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1` | Matched official Hugging Face LFS SHA-256 |
| Piper en_US-amy-medium | `models/tts/en_US-amy-medium/en_US-amy-medium.onnx` | 63,201,294 bytes (60.27 MiB) | `b3a6e47b57b8c7fbe6a0ce2518161a50f59a9cdd8a50835c02cb02bdd6206c18` | Matched official Hugging Face LFS SHA-256 |

The downloaded set consumes 257,840,357 bytes (245.90 MiB, 0.240 GiB), including required local metadata and companion files. Depth Anything also has `config.json` and `preprocessor_config.json`; faster-whisper tiny also has `config.json`, `tokenizer.json`, and `vocabulary.txt`; Piper has its `.onnx.json` companion and `MODEL_CARD`.

Phase 1C added the YOLOE text-prompt encoder required by the documented text workflow:

- Local path: `models/detection/yoloe-text/mobileclip2_b.ts`
- Size: 253,794,476 bytes (242.00 MiB)
- SHA-256: `35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f`
- Official source checksum: matched the Ultralytics v8.4.0 release metadata
- Offline inference test: blocked because `ultralytics` is not installed and is not declared in `requirements.txt`; no network inference test was run.

## A. Continuous object detection

### YOLO26n

- **Model name:** `yolo26n.pt`
- **Family/capability:** Ultralytics YOLO26, real-time closed-set object detection
- **Official source:** https://docs.ultralytics.com/models/yolo26/
- **License:** Ultralytics source and models are documented under AGPL-3.0 and Enterprise licensing options; deployment terms require review.
- **Parameters:** 2.4M in the official fused detection configuration
- **Weight size:** Not documented by the official model page
- **Runtime:** Official page reports 640px detection benchmarks and supports PyTorch/Ultralytics inference; this machine's CUDA runtime is not ready, so CPU is the only currently verified device.
- **Offline:** Yes after the checkpoint and runtime assets are present locally. Loading by model name may trigger a download and is forbidden in this phase.
- **FRIDAY role:** Always-on scene object proposals for navigation and safety reasoning.
- **Mode:** Continuous.
- **Limitations:** COCO-pretrained closed-set vocabulary; measure small-object and low-light performance locally.
- **Downloaded path:** `models/detection/yolo26n.pt`; verified by file existence, size, and local SHA-256 above.

### YOLO26s

- **Model name:** `yolo26s.pt`
- **Family/capability:** Ultralytics YOLO26, real-time closed-set object detection
- **Official source:** https://docs.ultralytics.com/models/yolo26/
- **License:** AGPL-3.0 or Ultralytics Enterprise terms.
- **Parameters:** 9.5M in the official fused detection configuration
- **Weight size:** Not documented by the official model page
- **Runtime:** Same runtime path as YOLO26n; the official table reports 640px detection metrics. Verify CPU latency and memory before considering continuous use.
- **Offline:** Yes with local checkpoint and runtime assets.
- **FRIDAY role:** Accuracy/latency comparison against YOLO26n for the continuous detector.
- **Mode:** Continuous.
- **Limitations:** More compute than nano; CUDA performance cannot yet be measured.

## B. Open-vocabulary object search

### YOLOE-26n-seg

- **Model name:** `yoloe-26n-seg.pt`
- **Family/capability:** Ultralytics YOLOE-26 text/visual-prompt open-vocabulary detection and instance segmentation
- **Official source:** https://docs.ultralytics.com/models/yoloe/
- **License:** AGPL-3.0 or Ultralytics Enterprise terms.
- **Parameters:** 3.9M for the official detection configuration; full released segmentation checkpoint count is not documented here.
- **Weight size:** Not documented by the official model page.
- **Runtime:** Supports text prompts through `set_classes`; YOLOE-26 text prompting fetches a `mobileclip2_b.ts` text encoder of about 254 MB on first use.
- **Offline:** Yes only after checkpoint, text encoder, tokenizer/runtime, and prompt assets are staged locally. Text prompting must not be assumed offline on first use.
- **FRIDAY role:** Low-cost command-driven search for “find a chair” or “where is the backpack?”.
- **Mode:** Search, not continuous default operation.
- **Limitations:** Zero-shot accuracy is below a task-trained closed-set detector; rare categories and large prompt sets can reduce quality or latency.
- **Downloaded path:** `models/detection/yoloe-26n-seg.pt`; verified by file existence, size, and local SHA-256 above.
- **First-use network dependency:** Text prompting may fetch and install the `ultralytics/CLIP` tokenizer dependency and download `mobileclip2_b.ts` (about 254 MB) into the working directory. That text encoder was intentionally not downloaded in Phase 1B. Visual-prompt and prompt-free workflows do not require that text encoder, according to the official documentation.

### YOLOE-26s-seg

- **Model name:** `yoloe-26s-seg.pt`
- **Family/capability:** Ultralytics YOLOE-26 text/visual-prompt open-vocabulary detection and instance segmentation
- **Official source:** https://docs.ultralytics.com/models/yoloe/
- **License:** AGPL-3.0 or Ultralytics Enterprise terms.
- **Parameters:** 10.7M for the official detection configuration; full released segmentation checkpoint count is not documented here.
- **Weight size:** Not documented by the official model page.
- **Runtime:** Same prompt/text-encoder requirements as YOLOE-26n; official deployment notes call for approximately 4-8 GB VRAM for GPU inference, with nano/small scales also usable on CPU at reduced resolution.
- **Offline:** Yes after all checkpoint and text-prompt assets are staged locally.
- **FRIDAY role:** Higher-quality search comparison for named personal objects such as a bottle or backpack.
- **Mode:** Search.
- **Limitations:** Approximately 6 GB VRAM is a constraint; CUDA is currently unavailable and full segmentation checkpoint size is not documented.

## C. Depth estimation

### Depth Anything V2 Small

- **Model name:** Official Transformers checkpoint `Depth-Anything-V2-Small-hf`
- **Family/capability:** Depth Anything V2 monocular relative depth estimation
- **Official source:** https://github.com/DepthAnything/Depth-Anything-V2
- **License:** Apache-2.0 for Small
- **Parameters:** 24.8M
- **Weight size:** 99,173,660 bytes (94.58 MiB) for the downloaded `model.safetensors`
- **Runtime:** Official example uses PyTorch and a default input size of 518; CPU/GPU device selection is supported by the example.
- **Offline:** Yes after the checkpoint and code/runtime are local.
- **FRIDAY role:** Fast approximate depth cues for obstacle distance ordering and scene layout.
- **Mode:** Continuous support, subject to frame-rate measurement.
- **Limitations:** Relative depth is not calibrated metric distance; do not use it as the sole safety signal.
- **Downloaded path:** `models/depth/depth-anything-v2-small/model.safetensors`, with local `config.json` and `preprocessor_config.json`; verified by file existence, size, and official LFS SHA-256.

### Depth Anything V2 Base

- **Model name:** `depth_anything_v2_vitb.pth` or the corresponding official Transformers Base checkpoint
- **Family/capability:** Depth Anything V2 monocular relative depth estimation
- **Official source:** https://github.com/DepthAnything/Depth-Anything-V2
- **License:** CC-BY-NC-4.0 for Base
- **Parameters:** 97.5M
- **Weight size:** Not documented by the official repository
- **Runtime:** Same 518px PyTorch inference path; benchmark separately because the model is substantially larger than Small.
- **Offline:** Yes after local staging.
- **FRIDAY role:** Accuracy comparison for difficult scenes where Small depth ordering is insufficient.
- **Mode:** Continuous support, but not assumed suitable until measured.
- **Limitations:** Higher memory and latency; excluded from always-on selection until benchmarked. Large is intentionally not a Phase 1A candidate because of the VRAM constraint.

## D. Speech-to-text

### faster-whisper small

- **Model name:** `Systran/faster-whisper-small` when materialized as a local CTranslate2 directory
- **Family/capability:** faster-whisper CTranslate2 conversion of Whisper Small
- **Official source:** https://huggingface.co/Systran/faster-whisper-small and https://github.com/SYSTRAN/faster-whisper
- **License:** MIT for the faster-whisper repository/model card; original Whisper model terms require separate review.
- **Parameters:** Not documented by the selected official conversion card
- **Weight size:** Not documented by the selected official conversion card
- **Runtime:** CPU `int8` and GPU `float16`/`int8_float16` are documented; current GPU path requires CUDA 12 cuBLAS and cuDNN 9.
- **Offline:** Yes when loaded from a local directory; loading by size/name may download from Hugging Face.
- **FRIDAY role:** Primary local command transcription, including “find”, “stop”, and navigation commands.
- **Mode:** On demand or short command stream.
- **Limitations:** Measure command word error rate and end-to-end capture-to-intent latency; current environment only proves CPU PyTorch.

### faster-whisper tiny

- **Model name:** `Systran/faster-whisper-tiny` when materialized locally
- **Family/capability:** faster-whisper CTranslate2 conversion of Whisper Tiny
- **Official source:** https://huggingface.co/Systran/faster-whisper-tiny and https://github.com/SYSTRAN/faster-whisper
- **License:** MIT for the faster-whisper repository/model card; review original Whisper terms separately.
- **Parameters:** Not documented by the selected official conversion card
- **Weight size:** Not documented by the selected official conversion card
- **Runtime:** CPU `int8` and GPU modes follow faster-whisper runtime requirements.
- **Offline:** Yes from a local model directory.
- **FRIDAY role:** Low-latency fallback and baseline for short commands.
- **Mode:** On demand or short command stream.
- **Limitations:** Expected accuracy tradeoff against Small must be measured rather than assumed.
- **Downloaded path:** `models/speech/faster-whisper-tiny/`, containing `model.bin`, `config.json`, `tokenizer.json`, and `vocabulary.txt`; verified by file existence, sizes, and official LFS SHA-256 for `model.bin`.

## E. Text-to-speech

### en_US-lessac-medium

- **Model name:** `en_US-lessac-medium.onnx` plus its `.onnx.json` file
- **Family/capability:** Piper VITS ONNX local speech synthesis
- **Official source:** https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md
- **License:** Voice-specific; Piper requires reviewing the voice's `MODEL_CARD` before redistribution.
- **Parameters:** Official Piper samples document 15-20M for the medium quality band.
- **Weight size:** Not documented by the official voice page.
- **Runtime:** ONNX Runtime local synthesis; medium quality is 22.05 kHz according to the official samples page.
- **Offline:** Yes after both local voice files are present.
- **FRIDAY role:** Primary English voice candidate for concise safety and navigation prompts.
- **Mode:** On demand.
- **Limitations:** Voice license and Windows audio output behavior require verification.

### en_US-amy-medium

- **Model name:** `en_US-amy-medium.onnx` plus its `.onnx.json` file
- **Family/capability:** Piper VITS ONNX local speech synthesis
- **Official source:** https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md
- **License:** Voice-specific; review its `MODEL_CARD`.
- **Parameters:** 15-20M quality-band range documented for Piper medium voices.
- **Weight size:** Not documented by the official voice page.
- **Runtime:** ONNX Runtime local synthesis at the medium quality band’s documented 22.05 kHz.
- **Offline:** Yes after both local voice files are present.
- **FRIDAY role:** Voice comparison for intelligibility and user preference.
- **Mode:** On demand.
- **Limitations:** Exact per-voice license and file size must be verified when the voice is deliberately acquired.
- **Downloaded path:** `models/tts/en_US-amy-medium/en_US-amy-medium.onnx` plus `en_US-amy-medium.onnx.json` and `MODEL_CARD`; verified by file existence, sizes, and official LFS SHA-256 for the ONNX file.

## F. Optional local LLM

These candidates are optional and are not part of FRIDAY safety logic. No weights are being downloaded.

### Qwen3-1.7B

- **Model name:** `Qwen/Qwen3-1.7B`
- **Family/capability:** Qwen3 causal language model
- **Official source:** https://huggingface.co/Qwen/Qwen3-1.7B
- **License:** Apache-2.0
- **Parameters:** 1.7B
- **Repository size:** 4.08 GB as shown by the official model repository; this is model repository size, not runtime RAM or VRAM.
- **Runtime:** Official card supports Transformers and local runtimes; actual quantized memory and latency require measurement.
- **Offline:** Yes after local staging.
- **FRIDAY role:** Optional dialogue/intent explanation helper only; never the independent safety authority.
- **Mode:** On demand.
- **Limitations:** 16 GB system RAM may support a quantized configuration, but this is unverified.

### Phi-4-mini-instruct

- **Model name:** `microsoft/Phi-4-mini-instruct`
- **Family/capability:** Phi-4 small instruction-following language model
- **Official source:** https://huggingface.co/microsoft/Phi-4-mini-instruct
- **License:** MIT
- **Parameters:** 3.8B
- **Repository size:** 7.69 GB as shown by the official model repository; this is not runtime RAM or VRAM.
- **Runtime:** Official card documents Transformers and vLLM paths; tested hardware is substantially larger than this laptop, so local feasibility is unverified.
- **Offline:** Yes after local staging.
- **FRIDAY role:** Optional natural-language response helper, isolated from perception and safety decisions.
- **Mode:** On demand.
- **Limitations:** 7.69 GB repository size plus runtime overhead is a serious constraint for approximately 15.6 GB RAM; benchmark only after core pipeline work.

## Download and verification gate

Before Phase 1 execution, verify each local checkpoint path, file hash, exact license, model file size, runtime RAM, runtime VRAM, and device. A registry entry is not evidence that a weight has been downloaded.
