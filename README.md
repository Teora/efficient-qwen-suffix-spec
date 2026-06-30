# Efficient Qwen Challenge — 2nd place (7.708×)

**Team AFM-6j6duhm6** · "Efficient Qwen" Competition, AdaptFM Workshop @ ICML 2026

A reproducible pipeline that accelerates **Qwen3.5-4B** inference **7.708×** over the
unoptimized baseline on a single NVIDIA A10G, while passing all three quality gates
(MMLU-Pro, IFEval, GPQA-Diamond). Final standing: **2nd of 40+ teams**, 0.037× behind 1st.

- Workshop: [Resource-Adaptive Foundation Model Inference (AdaptFM) @ ICML 2026](https://adaptfm.gitlab.io/)
- Competition: [Efficient Qwen — Minimizing Inference Latency for Qwen3.5-4B on A10G](https://adaptfm.gitlab.io/call-for-competition.html)
- Leaderboard: https://d1krc5fcnf73gi.cloudfront.net/

## The recipe

| Component | What | Why |
|---|---|---|
| **Quantization** | AWQ-INT4 on **MLP only** (attn / layer 0 / MTP head / vision = FP16), `awq_marlin` kernel | INT4 MLP cuts the dominant decode cost; keeping attn FP16 preserves quality |
| **Vocab pruning** | 248,320 → **131,101** tokens, `prune_vocab_awq.py` | Shrinks the tied embed/lm_head ~1.9×, the biggest per-token decode memory cost. Faster decode/verify, GPQA gate preserved |
| **Speculative decoding** | **Suffix always-spec**, `K=20`, `max_spec_factor=3.0`, tree depth 64 (arctic-inference) | Caches previous responses globally → drafts repeated spans for free; **lossless** (verification-exact) |
| **Force-answer wrapper** | `serve.py`: on a thinking response that hits the token budget without closing `</think>`, append a content-blind bridge so the model self-finishes its answer | Recovers long-reasoning GPQA samples that would otherwise truncate |
| **Repetition penalty** | `1.15`, **thinking-path only** (`serve.py`, keyed off the grader's `enable_thinking` flag) | Stops greedy long-reasoning loops on GPQA; no effect on the latency path or MMLU/IFEval |
| **Engine** | vLLM 0.19.0, bf16, prefix-caching, CUDA graphs | Competition base image |

### Key finding: `max_spec_factor` is the quality lever
Suffix speculation is *verification-exact* (lossless) in principle, but under temperature
sampling + concurrency the **aggressiveness factor controls quality**:
- **`factor = 3.0` → PASSING regime** (our submission). Hidden GPQA clears the gate.
- `factor ≥ 4` → long reasoning **degenerates** (rambling, unclosed `</think>`) and fails
  the hidden GPQA gate, even though local/short-output quality looks fine.

We measured this directly: `K=20/factor5` scored hidden GPQA **0.606 (FAIL)** twice
(deterministically), while `K=20/factor3` **passed** at the same speed tier. The winning
move was to back off the factor, not the speculation depth.

### How the vocab is pruned (`prune_vocab_awq.py`)
The original tied embed/lm_head (248,320 tokens, BF16, **not** AWQ-quantized) is the
single largest per-token decode cost (~1.27 GB loaded every step). Pruning is purely
**ID-order based**, not frequency- or eval-driven:

1. **Keep regular tokens `[0, KEEP)`** (KEEP = 131,072) and drop the high-ID tail.
2. **Keep all special/added tokens**, remapped contiguously to `[KEEP, KEEP+n)` →
   final vocab **131,101**.
3. **Slice the embedding row-wise** to match; AWQ transformer blocks, the MTP head,
   and (optionally) the vision tower pass through untouched.
4. **Rebuild BPE merges** keeping only rules whose components both survive, and
   **remap every special token ID** in `config` / `generation_config`
   (critically `eos_token_id` — a stale value never stops generation and blows up latency).

This *reduces* parameters (it cannot exceed the original count) and is independent of any
eval set. `v96k` pruning fails GPQA; **`v128k` (131k) is the smallest vocab that preserves
the GPQA-Diamond (thinking) gate.**

## Repository layout

```
Dockerfile              # self-contained build of the exact submitted image
serve.py                # :8080 -> vLLM :8081 proxy (thinking detection, force-answer, rep-penalty)
prune_vocab_awq.py      # vocab pruning 248320 -> 131101 on the AWQ checkpoint
engines/
  wheels/arctic_inference-0.1.1-...whl          # Apache-2.0 suffix-decoding backend
  patches/suffix_decoding_v019_alwaysspec.py    # 27-line lossless always-spec patch (Apache-2.0)
bench.py                # latency benchmark (short/medium/long, vs baseline)
eval_local.py           # quality harness (MMLU-Pro / IFEval / GPQA-Diamond, conc=8)
run_gpqa.sh             # GPQA-Diamond convenience runner
MODEL_CARD.md           # Hugging Face model card for the released weights
```

Model weights (`qwen-awq-v128k/`, ~5.2 GB) are released separately on the Hugging Face
Hub ([Teora/qwen3.5-4b-awq-v128k](https://huggingface.co/Teora/qwen3.5-4b-awq-v128k)) — too large for git.

## Reproduce

### 0. Prerequisites
- 1× NVIDIA L4 or A10G (24 GB), Docker with NVIDIA runtime
- The competition base image `adaptfm/adaptfm-base:latest` (vLLM 0.19.0)

### 1. Get the weights
Either pull the released AWQ-v128k checkpoint from the HF Hub into `./qwen-awq-v128k/`,
**or** rebuild it from the AWX-INT4 source:
```bash
python prune_vocab_awq.py --src qwen-awq-int4 --out qwen-awq-v128k --keep 131072
```

### 2. Build
```bash
docker build -f Dockerfile -t efficient-qwen-2nd:latest .
```

### 3. Serve
```bash
docker run -d --gpus '"device=0"' -p 8080:8080 --name eq2 efficient-qwen-2nd:latest
# endpoints: /ping  /invocations  /v1/completions  /v1/chat/completions
curl -s localhost:8080/ping
```

### 4. Benchmark speed (reproduces ~7.7×)
```bash
CONTAINER_URL=http://localhost:8080 python bench.py --tag eq2 --runs 50
# baseline (A10G): short 2582ms / medium 5441ms / long 6576ms
# ours    (A10G): ~240 / ~620 / ~1833ms  ->  ~7.7x average
```

### 5. Verify quality gates
```bash
# GPQA-Diamond (thinking, conc=8) — the binding gate
NUM_CONCURRENT=8 bash run_gpqa.sh http://localhost:8080 results/gpqa.txt eq2 198
# full suite (MMLU-Pro / IFEval / GPQA)
EVAL_MODE=full CONTAINER_URL=http://localhost:8080 python eval_local.py
# gates: MMLU-Pro >= 0.621 · IFEval >= 0.814 · GPQA-Diamond >= 0.630
```

## Results

| Metric | Baseline | Ours | Speedup |
|---|---|---|---|
| Short (64→128 tok) | 2,582 ms | ~240 ms | — |
| Medium (2048→256) | 5,441 ms | ~620 ms | — |
| Long (8192→256) | 6,576 ms | ~1,833 ms | — |
| **Average** | 4,866 ms | — | **7.708×** |
| MMLU-Pro | — | PASS | ≥0.621 |
| IFEval | — | PASS | ≥0.814 |
| GPQA-Diamond | — | PASS | ≥0.630 |

## Compliance
No prohibited techniques: no hard-coded/cached answers, no benchmark detection/routing,
no obfuscation, no training on eval data. Vocab pruning *reduces* parameters (it cannot
exceed the original count). Thinking-mode gating keys off the grader's own
`enable_thinking` flag — not content detection. Force-answer injects no answer content,
only a bridge token that closes the think block. The suffix patch is plain, commented,
Apache-2.0 Python.

## License
**Apache-2.0** (this code and the released weights). Qwen3.5-4B is Apache-2.0;
arctic-inference is Apache-2.0. See `LICENSE`.

## Team
Minjae Park · CJ Corporation · minjaepark900@gmail.com

## Citation
Minjae Park (CJ Corporation). "Suffix always-spec decoding for efficient Qwen3.5-4B
inference." 2nd place, Efficient Qwen Competition, AdaptFM Workshop @ ICML 2026.
https://github.com/Teora/efficient-qwen-suffix-spec
