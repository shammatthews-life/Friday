# Qwen3-8B RTX 4050 Vulkan GPU-layer sweep

- Result: **PASS**
- Model: Qwen3-8B Q4_K_M
- Runtime: llama.cpp `0.5.0-dev` build 11146
- Selected GPU: NVIDIA GeForce RTX 4050 Laptop GPU (`Vulkan1`); AMD Radeon 740M is `Vulkan0` and was not selected.
- Fixed settings: context 4096; batch/ubatch 128/32; 8 CPU threads; flash attention auto; default KV cache.
- Repetitions per configuration: 2
- Tested GPU layers: 28, 30, 32, 36
- Test order: 28, 30, 32, 32, 30, 28; optional 36-layer check only after the 32-layer stability/headroom gate.
- Conditional next-layer decision: Eligible: projected 36-layer peak 4934 MiB leaves 1207 MiB headroom.

All values below are arithmetic means across the two full four-turn runs; only the three visual turns are averaged for the main latency comparison. Per-turn values are shown separately.

## Configuration summary

| GPU layers | Startup (s) | Prompt tokens | Prefill (s) | First token (s) | Completion (s) | Total response (s) | Generated tokens | tok/s | Peak process RSS (MiB) | Peak RTX VRAM (MiB) | Mean GPU util (%) | Stability |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 28 | 10.30 | 2252.50 | 22.20 | 22.39 | 3.68 | 26.07 | 33.00 | 9.16 | 6165.40 | 3952.00 | 22.45 | PASS |
| 30 | 7.48 | 2261.33 | 19.27 | 19.47 | 3.64 | 23.11 | 38.67 | 10.76 | 6109.55 | 4196.00 | 26.84 | PASS |
| 32 | 8.51 | 2241.00 | 15.71 | 15.90 | 1.81 | 17.71 | 25.67 | 14.18 | 6098.25 | 4443.00 | 28.89 | PASS |
| 36 | 8.49 | 2241.00 | 8.66 | 8.81 | 1.19 | 10.00 | 25.67 | 21.61 | 6038.15 | 4954.00 | 43.60 | PASS |

## Change from 28 layers

| GPU layers | First-token change | Total-response change | Prefill change | tok/s change | Peak RTX VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|
| 28 | +0.0% | +0.0% | +0.0% | +0.0% | 3952.0 |
| 30 | -13.0% | -11.3% | -13.2% | +17.5% | 4196.0 |
| 32 | -29.0% | -32.1% | -29.2% | +54.8% | 4443.0 |
| 36 | -60.6% | -61.6% | -61.0% | +135.9% | 4954.0 |

## Same-turn results

| GPU layers | Question | Prompt tokens | Prefill (s) | First token (s) | Completion (s) | Total (s) | Generated tokens | tok/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 28 | What happened in this video? | 2193.00 | 24.91 | 24.94 | 5.00 | 29.94 | 43.00 | 8.76 |
| 28 | What about the sports ball? | 2258.00 | 20.86 | 21.11 | 3.92 | 25.03 | 35.50 | 9.07 |
| 28 | When did it appear? | 2306.50 | 20.85 | 21.11 | 2.12 | 23.24 | 20.50 | 9.65 |
| 28 | What text was shown? | 1053.00 | 8.44 | 8.73 | 1.62 | 10.35 | 17.00 | 10.53 |
| 30 | What happened in this video? | 2193.00 | 21.72 | 21.74 | 5.01 | 26.76 | 53.00 | 10.59 |
| 30 | What about the sports ball? | 2268.00 | 18.00 | 18.24 | 4.03 | 22.28 | 42.00 | 10.48 |
| 30 | When did it appear? | 2323.00 | 18.10 | 18.43 | 1.87 | 20.30 | 21.00 | 11.21 |
| 30 | What text was shown? | 1070.00 | 5.54 | 5.82 | 1.37 | 7.19 | 17.00 | 12.41 |
| 32 | What happened in this video? | 2193.00 | 18.12 | 18.15 | 2.21 | 20.35 | 31.00 | 14.06 |
| 32 | What about the sports ball? | 2246.00 | 14.45 | 14.69 | 1.75 | 16.44 | 25.00 | 14.29 |
| 32 | When did it appear? | 2284.00 | 14.57 | 14.87 | 1.48 | 16.35 | 21.00 | 14.19 |
| 32 | What text was shown? | 1031.00 | 4.60 | 4.87 | 1.11 | 5.98 | 17.00 | 15.33 |
| 36 | What happened in this video? | 2193.00 | 9.58 | 9.61 | 1.48 | 11.09 | 31.00 | 20.98 |
| 36 | What about the sports ball? | 2246.00 | 8.26 | 8.49 | 1.11 | 9.60 | 25.00 | 22.57 |
| 36 | When did it appear? | 2284.00 | 8.14 | 8.34 | 0.99 | 9.33 | 21.00 | 21.28 |
| 36 | What text was shown? | 1031.00 | 2.46 | 2.72 | 0.76 | 3.48 | 17.00 | 22.37 |

## Per-run stability

- 28 layers, run 1: PASS; grounding PASS; peak RSS 6176.3 MiB; peak RTX VRAM 3952.0 / 6141 MiB; GPU samples 179.
- 28 layers, run 2: PASS; grounding PASS; peak RSS 6154.5 MiB; peak RTX VRAM 3952.0 / 6141 MiB; GPU samples 156.
- 30 layers, run 1: PASS; grounding PASS; peak RSS 6079.7 MiB; peak RTX VRAM 4191.0 / 6141 MiB; GPU samples 150.
- 30 layers, run 2: PASS; grounding PASS; peak RSS 6139.4 MiB; peak RTX VRAM 4196.0 / 6141 MiB; GPU samples 148.
- 32 layers, run 1: PASS; grounding PASS; peak RSS 6097.9 MiB; peak RTX VRAM 4443.0 / 6141 MiB; GPU samples 119.
- 32 layers, run 2: PASS; grounding PASS; peak RSS 6098.6 MiB; peak RTX VRAM 4443.0 / 6141 MiB; GPU samples 118.
- 36 layers, run 1: PASS; grounding PASS; peak RSS 6040.6 MiB; peak RTX VRAM 4954.0 / 6141 MiB; GPU samples 84.
- 36 layers, run 2: PASS; grounding PASS; peak RSS 6035.7 MiB; peak RTX VRAM 4954.0 / 6141 MiB; GPU samples 84.

## Decision

- Best measured setting: 36 GPU layers, ranked by mean visual first-token latency then total response among stable configurations.
- Safest tested setting: 28 GPU layers (lowest tested offload with all runs stable).
- Best vs. 28 layers: 60.6% lower mean visual first-token latency.
- Approximate RTX VRAM headroom at best setting: 1187 MiB.
- Peak process RSS at best setting: 6038 MiB. Against 15.2 GiB installed system RAM, this leaves about 9.3 GiB nominally beyond the measured process tree; actual free-RAM headroom during inference was not sampled and will be lower after accounting for Windows and other processes.
- Peak process RSS and VRAM are sampled maxima, not reserved-memory guarantees.
- The sweep runner now includes the GPU-layer value correctly in its generated configuration table; startup values above were retained in the original run output.
- 28 → 30 layers: first token -13.0%, total response -11.3% (clear improvement).
- 30 → 32 layers: first token -18.3%, total response -23.4% (clear improvement).
- 32 → 36 layers: first token -44.6%, total response -43.5% (clear improvement).
- Plateau/regression thresholds: changes within ±5% on both primary latencies count as plateau; a ≥5% increase in either counts as regression.
- Production/default GPU layers were not changed.

## Failures

- None
