# =============================================================================
# AdaptFM "Efficient Qwen" Challenge — 2nd place (7.708x)  ·  Team AFM-6j6duhm6
# Winning card: SUFFIX always-spec decoding (K=20, max_spec_factor=3.0) on an
#               AWQ-INT4(MLP) + v128k-vocab-pruned Qwen3.5-4B, with a content-blind
#               force-answer wrapper for long reasoning.
#
# Self-contained reproduction of the exact submitted image
#   (originally built as a 4-stage chain: mtp12-pc -> suffixpad -> suffix8 ->
#    suffix8-fa -> suffix20f3; flattened here into one Dockerfile).
#
# Build:
#   docker build -f release/Dockerfile -t efficient-qwen-2nd:latest .
# Run (single L4/A10G, port 8080):
#   docker run -d --gpus '"device=0"' -p 8080:8080 efficient-qwen-2nd:latest
# Endpoints: /ping  /invocations  /v1/completions  /v1/chat/completions
# =============================================================================

# competition base image (vLLM 0.19.0, CUDA 12.4)
FROM adaptfm/adaptfm-base:latest

ENV TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/program

# --- Model weights -----------------------------------------------------------
# AWQ-INT4 on MLP only (attn / layer0 / vision = FP16); vocab pruned 248320 -> 131101
# (keep contiguous token IDs [0,131072) + specials). Pruning shrinks lm_head ~1.9x,
# which speeds decode/verify while preserving the GPQA(thinking) quality gate.
# Build the weights yourself with prune_vocab_awq.py, or pull from the HF release.
COPY qwen-awq-v128k/ /opt/ml/model/

# --- Suffix always-spec engine (lossless) ------------------------------------
# arctic-inference (Apache-2.0) provides SuffixDecodingCache; the 27-line patch
# makes it lossless by padding empty drafts to one real token so every decode step
# stays on the spec path, and rejection sampling guarantees bit-identical output.
COPY engines/wheels/arctic_inference-0.1.1-cp310-cp310-linux_x86_64.whl /tmp/
RUN pip install --no-deps /tmp/arctic_inference-0.1.1-cp310-cp310-linux_x86_64.whl && \
    rm /tmp/arctic_inference-0.1.1-cp310-cp310-linux_x86_64.whl
COPY engines/patches/suffix_decoding_v019_alwaysspec.py \
    /usr/local/lib/python3.10/dist-packages/vllm/v1/spec_decode/suffix_decoding.py

# --- Serving proxy -----------------------------------------------------------
# Thin :8080 -> vLLM :8081 proxy. Honors the grader's enable_thinking flag;
# applies thinking-only repetition_penalty; and force-answer: when a thinking
# response hits the token budget without closing </think>, it appends a
# content-blind bridge that closes the block and lets the model self-finish.
COPY serve.py /opt/program/serve_default.py

ENV MODEL_PATH=/opt/ml/model \
    SERVE_PORT=8080 \
    VLLM_PORT=8081 \
    VLLM_QUANTIZATION=awq_marlin \
    VLLM_LANGUAGE_MODEL_ONLY=1 \
    VLLM_DTYPE=bfloat16 \
    VLLM_GPU_MEMORY_UTILIZATION=0.90 \
    VLLM_MAX_MODEL_LEN=13312 \
    VLLM_ENABLE_PREFIX_CACHING=1

# --- Winning spec config: suffix, K=20, factor 3.0 ---------------------------
# max_spec_factor=3.0 is the PASSING regime: factor>=4 degenerates long GPQA
# reasoning (unclosed </think>, rambling) and fails the hidden gate; factor=3.0
# keeps quality while clearing #1's speed bar. Lossless (verification-exact).
ENV VLLM_SPEC_CONFIG='{"method":"suffix","num_speculative_tokens":20,"suffix_decoding_max_tree_depth":64,"suffix_decoding_max_spec_factor":3.0,"suffix_decoding_min_token_prob":0.0}'

# Force-answer wrapper ON (recovers GPQA samples that reason past the budget).
ENV VLLM_THINK_FORCE_ANSWER=1

# Thinking-only repetition penalty (stops greedy long-reasoning loops; no effect
# on the non-thinking latency path or on MMLU/IFEval).
ENV VLLM_CHAT_REP_PENALTY=1.15

# CUDA-graph capture sizes must be multiples of (num_speculative_tokens+1)=21.
ENV VLLM_CUDAGRAPH_CAPTURE_SIZES='21 42'
ENV VLLM_EXTRA_ARGS=

EXPOSE 8080
