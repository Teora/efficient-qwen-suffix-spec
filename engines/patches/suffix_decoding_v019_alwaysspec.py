# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch


class SuffixDecodingProposer:
    """
    Speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).
    This class imports and uses the official implementation from Arctic Inference
    (https://github.com/snowflakedb/ArcticInference).
    """

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len

        # Lazy import to avoid error when Suffix Decoding is not used.
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        # Initialize and empty cache. This object will take care of caching request
        # outputs, evicting old requests, and manages the per-prompt suffix trees.
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=config.suffix_decoding_max_cached_requests,
        )

    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        """
        Propose speculative tokens for each request in the input batch. Suffix Decoding
        will speculate a dynamic number of tokens for each request every decoding step,
        so each entry in the returned list may have different lengths.
        """
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]
            if req_id not in self.suffix_cache.active_requests:
                if req_id in self.suffix_cache.cached_requests:
                    # Reset the suffix cache for this request.
                    self.suffix_cache.evict_cached_response(req_id)
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                # Start a new request, this will build the suffix tree for that prompt.
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # Append the newly sampled ids to the suffix cache for this request.
            self.suffix_cache.add_active_response(req_id, sampled_ids)

            # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
            # we extract the pattern from the end of the input.
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            # ALWAYS-SPEC CONSTANT-K PATCH (2026-06-10): always return EXACTLY
            # max_spec tokens for every decode step. Two bugs are closed at once:
            # (1) quality corruption (#39273): an empty draft drops the request off the
            #     spec path next step, and the GDN backend then reads/writes recurrent
            #     state at slot 0 with a stale value (suffix GPQA was 0.27-0.53). Any
            #     non-empty draft keeps the multi-slot per-token state checkpointing
            #     self-consistent — the mechanism that makes MTP/ngram_gpu lossless.
            # (2) EngineDead (#39809): VARIABLE spec lengths + prefix caching drift the
            #     hybrid KV block accounting ('num_required_blocks 65 < len(req_blocks)
            #     66' assert assumes the lookahead never shrinks). ngram_gpu survives
            #     identical load because its drafts are always exactly k long — so we
            #     pad to constant k as well.
            # Pad tokens are only ACCEPTED if they match the verifier's greedy choice,
            # so correctness is guaranteed by rejection sampling. Wasted verify tokens
            # are ~free in the memory-bound decode regime (ngram_gpu ns24 GPQA at c=8
            # ran ~34s/sample with the same constant-k verify cost).
            # PAD FILL = -1, mirroring ngram_gpu's proven end-to-end behavior. Filling
            # with REAL repeated tokens polluted the repetition-penalty context on the
            # thinking path and cratered GPQA to 0.3 (2026-06-10 measured); -1 pads are
            # effectively inert there (ngram_gpu ships -1 through the same scheduler
            # path and holds GPQA ~0.556-0.65 class).
            max_spec = min(
                self.num_speculative_tokens, self.max_model_len - num_tokens - 1
            )
            ids = list(draft.token_ids)[:max_spec]
            if len(ids) < max_spec:
                ids = ids + [-1] * (max_spec - len(ids))
            draft_token_ids.append(ids)

        # Stop requests that were not seen in the input batch.
        for req_id in (
            self.suffix_cache.active_requests - input_batch.req_id_to_index.keys()
        ):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass
