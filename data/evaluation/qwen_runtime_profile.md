# Qwen3-8B runtime performance profile

- Result: **PASS**
- Device: NVIDIA GeForce RTX 4050 Laptop GPU; adapter selects its Vulkan device explicitly.
- Other enumerated device: AMD Radeon 740M Graphics (`Vulkan0`); Qwen selected `Vulkan1`.
- llama.cpp executable: `D:\friday\.venv\llama.cpp\llama-server.exe`
- llama.cpp build: `0.00.039.204 I srv  llama_server: initializing ...
version: 0.5.0-dev (build 11146, commit 7fe450e19)
built with Clang 20.1.8 for Windows x86_64`
- GPU: `Vulkan1`
- CPU: AMD Ryzen 7 7445HS (6 cores / 12 logical processors)
- CPU generation/prompt threads: 8
- Logical batch / physical ubatch: 128 / 32
- Context size: 4096
- Repetitions per configuration: 2
- Flash attention: auto (default; not specified)
- KV cache types: default/auto (not specified)
- Other flags: `{'parallel': 1, 'jinja': True, 'reasoning': 'off', 'webui': False, 'device': 'Vulkan1'}`

## Runtime startup diagnostics

```text
0.00.003.490 I srv  llama_server: initializing ...
0.01.135.499 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.01.141.267 I srv  init_listene: The UI is disabled
0.01.141.279 I srv  init_listene: Use --ui/--no-ui (or deprecated --webui/--no-webui) to enable/disable
0.01.142.712 W srv  llama_server: security: no API key is set and CORS allows all origins (see https://github.com/ggml-org/llama.cpp/pull/25655)
0.01.161.752 I srv    load_model: loading model 'D:\friday\models\llm\qwen3-8b\Qwen3-8B-Q4_K_M.gguf'
0.05.530.086 W load: control-looking token: 128247 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
0.15.330.414 I cmn          init: llama threadpool init, n_threads = 8
0.19.492.004 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'
0.19.545.855 I srv  llama_server: model loaded
0.19.546.254 I srv  llama_server: listening on http://127.0.0.1:65487
```

The measured first-token interval minus HTTP connection setup is an estimate of prompt/prefill time; llama-server does not expose an independent prefill timer in this adapter path.

## Configuration comparison

| Metric (mean over three visual turns and both runs) | Current (24 layers) | Test (28 layers) | Change |
|---|---:|---:|---:|
| Model/server startup (s) | 14.289 | 12.289 | -14.0% |
| Prompt/prefill (s) | 29.707 | 22.435 | -24.5% |
| First token (s) | 29.901 | 22.664 | -24.2% |
| Generation (s) | 4.051 | 3.781 | -6.7% |
| Total FRIDAY turn (s) | 33.953 | 26.446 | -22.1% |
| Prompt tokens | 2241.000 | 2252.500 | +0.5% |
| Completion tokens | 25.667 | 33.000 | +28.6% |
| Generation tokens/s | 6.430 | 8.864 | +37.8% |

## Same-turn comparison

| Turn | Metric | Current (24 layers) | Test (28 layers) | Change |
|---|---|---:|---:|---:|
| What happened in this video? | First token (s) | 33.678 | 25.811 | -23.4% |
| What happened in this video? | Prompt/prefill (s) | 33.633 | 25.747 | -23.4% |
| What happened in this video? | Generation (s) | 5.287 | 5.067 | -4.2% |
| What happened in this video? | Total response (s) | 38.966 | 30.878 | -20.8% |
| What happened in this video? | Prompt tokens | 2193.000 | 2193.000 | +0.0% |
| What happened in this video? | Completion tokens | 31.000 | 43.000 | +38.7% |
| What happened in this video? | Generation tokens/s | 5.890 | 8.514 | +44.6% |
| What about the sports ball? | First token (s) | 27.895 | 21.462 | -23.1% |
| What about the sports ball? | Prompt/prefill (s) | 27.624 | 21.150 | -23.4% |
| What about the sports ball? | Generation (s) | 3.699 | 4.072 | +10.1% |
| What about the sports ball? | Total response (s) | 31.595 | 25.534 | -19.2% |
| What about the sports ball? | Prompt tokens | 2246.000 | 2258.000 | +0.5% |
| What about the sports ball? | Completion tokens | 25.000 | 35.500 | +42.0% |
| What about the sports ball? | Generation tokens/s | 6.760 | 8.785 | +30.0% |
| When did it appear? | First token (s) | 28.128 | 20.718 | -26.3% |
| When did it appear? | Prompt/prefill (s) | 27.865 | 20.408 | -26.8% |
| When did it appear? | Generation (s) | 3.167 | 2.206 | -30.3% |
| When did it appear? | Total response (s) | 31.296 | 22.925 | -26.7% |
| When did it appear? | Prompt tokens | 2284.000 | 2306.500 | +1.0% |
| When did it appear? | Completion tokens | 21.000 | 20.500 | -2.4% |
| When did it appear? | Generation tokens/s | 6.640 | 9.292 | +39.9% |
| What text was shown? | First token (s) | 8.823 | 6.577 | -25.5% |
| What text was shown? | Prompt/prefill (s) | 8.529 | 6.277 | -26.4% |
| What text was shown? | Generation (s) | 2.225 | 1.623 | -27.1% |
| What text was shown? | Total response (s) | 11.049 | 8.200 | -25.8% |
| What text was shown? | Prompt tokens | 1031.000 | 1053.000 | +2.1% |
| What text was shown? | Completion tokens | 17.000 | 17.000 | +0.0% |
| What text was shown? | Generation tokens/s | 7.642 | 10.473 | +37.1% |

## Memory and GPU samples

| Configuration | Peak process-tree RSS (MiB) | Peak RTX 4050 memory used (MiB) | Mean RTX 4050 utilization (%) | GPU samples |
|---|---:|---:|---:|---:|
| Current: 24 GPU layers | 5282.2 | 3447.0 | 18.6 | 417 |
| Test: 28 GPU layers | 5536.5 | 3952.0 | 19.5 | 338 |

## Grounding and run validation

- Current: 24 GPU layers, run 1: smoke PASS; grounding checks PASS; RTX 4050 selected YES.
- Current: 24 GPU layers, run 2: smoke PASS; grounding checks PASS; RTX 4050 selected YES.
- Test: 28 GPU layers, run 1: smoke PASS; grounding checks PASS; RTX 4050 selected YES.
- Test: 28 GPU layers, run 2: smoke PASS; grounding checks PASS; RTX 4050 selected YES.

## Findings

- Prefill remains the dominant visual-turn cost; compare its measured change with generation time and repeat-to-repeat spread before attributing a gain to the GPU-layer change.
- Mean visual prefill: 29.707s → 22.435s; mean generation: 4.051s → 3.781s.
- Same four user turns and grounding checks were used in every run. Only `--n-gpu-layers` changed (24 → 28); no prompt, model, context, batch, thread, cache, or FRIDAY behavior changes were made for this comparison. Runs used a counterbalanced 24/28/28/24 order to reduce run-order/cache bias.

## Failures

- None
