---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
  - qwen
  - awq
  - int4
  - quantization
  - speculative-decoding
  - efficient-inference
  - vllm
language:
  - en
---

# Qwen3.5-4B — AWQ-INT4 (MLP) + v128k vocab pruning

Quantized & vocab-pruned **Qwen3.5-4B** used by team **AFM-6j6duhm6** to place
**2nd (7.708×)** in the AdaptFM "Efficient Qwen" Challenge (ICML 2026).

Pair this checkpoint with the reproducible serving pipeline (suffix always-spec
decoding, K=20, factor 3.0 + force-answer wrapper):
👉 **https://github.com/Teora/efficient-qwen-suffix-spec**

## What's modified vs the base model

| Change | Detail |
|---|---|
| **Quantization** | AWQ-INT4 on **MLP only** (attention, layer 0, MTP head, vision tower stay FP16). `awq_marlin` kernel. |
| **Vocab pruning** | Tied embed / lm_head pruned **248,320 → 131,101** tokens: keep contiguous IDs `[0,131072)` + all special/added tokens (remapped contiguously). This *reduces* parameters. |
| **MTP head** | Kept intact (native multi-token-prediction head preserved). |

Vocab pruning shrinks the BF16 tied embed/lm_head (~1.27 GB, the dominant per-token
decode memory cost) by ~1.9×, speeding decode/verify while preserving the
GPQA-Diamond (thinking) quality gate. v96k pruning failed GPQA; **v128k (131k) is
the minimum vocab that preserves it.**

## Quality (competition gates, all PASS)

| Benchmark | Threshold | Result |
|---|---|---|
| MMLU-Pro | ≥ 0.621 | PASS |
| IFEval | ≥ 0.814 | PASS |
| GPQA-Diamond (thinking) | ≥ 0.630 | PASS |

## Usage

```bash
# 1) download this checkpoint into ./qwen-awq-v128k/
huggingface-cli download Teora/qwen3.5-4b-awq-v128k --local-dir qwen-awq-v128k

# 2) build & serve via the pipeline repo
git clone https://github.com/Teora/efficient-qwen-suffix-spec && cd efficient-qwen-suffix-spec
docker build -f Dockerfile -t efficient-qwen-2nd:latest .
docker run -d --gpus '"device=0"' -p 8080:8080 efficient-qwen-2nd:latest
```

## License
**Apache-2.0** (Qwen3.5-4B is Apache-2.0; this derivative checkpoint inherits it).

## Citation
Team AFM-6j6duhm6 · Minjae Park (CJ Corporation) · AdaptFM Efficient Qwen Challenge, ICML 2026.
