#!/usr/bin/env python3
"""Vocab pruning for the AWQ Qwen3.5-4B checkpoint (keeps AWQ blocks + MTP intact).

The 248,320-token tied embed/lm_head is BF16 (not AWQ-quantized) and is the dominant
per-token DECODE memory cost (~1.27GB loaded every step). We shrink it to KEEP regular
tokens [0,KEEP) + ALL special/added tokens (remapped contiguously). AWQ-quantized
transformer blocks, the MTP head, and (optionally) the vision tower are copied unchanged.
Tied lm_head follows embed automatically.

Adapted from claude/aeq-icml/scripts/models/prune_vocab.py, but: (1) source = AWQ model,
(2) MTP KEPT (mtp_num_hidden_layers unchanged) so qwen3_5_mtp stays available,
(3) AWQ quant tensors passed through untouched.

Usage: python prune_vocab_awq.py --src qwen-awq-int4 --out qwen-awq-v<KEEP> --keep 32768 [--drop-visual]
"""
import argparse, json, os, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="qwen-awq-int4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=32768)
    ap.add_argument("--drop-visual", action="store_true", help="also drop vision tower weights (text-only)")
    args = ap.parse_args()
    SRC, OUT, KEEP = args.src, args.out, args.keep
    os.makedirs(OUT, exist_ok=True)

    # --- tokenizer.json: keep regular [0,KEEP), remap specials to [KEEP, KEEP+n) ---
    tj = json.load(open(f"{SRC}/tokenizer.json"))
    added = tj.get("added_tokens", [])
    sp_old = sorted(a["id"] for a in added)
    sp_base = sp_old[0]
    remap_sp = {old: KEEP + i for i, old in enumerate(sp_old)}
    keep_vocab = {tok: i for tok, i in tj["model"]["vocab"].items() if i < KEEP}
    tj["model"]["vocab"] = keep_vocab
    kept = set(keep_vocab)
    def _merge_ok(m):
        # BOTH components must also survive: merge results can have lower ids than
        # their components (caused 'Token ... out of vocabulary' BPE init failures
        # for KEEP>131072 — the historical "v160k+ 빌드불가").
        p = m.split(" ") if isinstance(m, str) else m
        return len(p) == 2 and p[0] in kept and p[1] in kept and (p[0] + p[1]) in kept
    tj["model"]["merges"] = [m for m in tj["model"].get("merges", []) if _merge_ok(m)]
    for a in added:
        a["id"] = remap_sp[a["id"]]
    tj["added_tokens"] = added
    json.dump(tj, open(f"{OUT}/tokenizer.json", "w"))
    tc = json.load(open(f"{SRC}/tokenizer_config.json"))
    if "added_tokens_decoder" in tc:
        tc["added_tokens_decoder"] = {str(remap_sp[int(k)]): v for k, v in tc["added_tokens_decoder"].items()
                                      if int(k) in remap_sp}
    json.dump(tc, open(f"{OUT}/tokenizer_config.json", "w"))
    # NOTE: merges.txt/vocab.json are NOT copied — they'd be stale full-vocab GPT2-style
    # duplicates that the slow-tokenizer fallback reads, breaking BPE init after pruning.
    for fn in ["chat_template.jinja", "generation_config.json",
               "special_tokens_map.json", "preprocessor_config.json"]:
        if os.path.isfile(f"{SRC}/{fn}"):
            shutil.copy2(f"{SRC}/{fn}", f"{OUT}/{fn}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(OUT)
    n_total = max(tok.vocab.values()) + 1
    n_top = n_total - KEEP
    print(f"keep {KEEP} regular + {n_top} top(specials) -> vocab_size {n_total}", flush=True)

    # --- weights: slice embed, pass AWQ blocks/MTP through, optionally drop visual ---
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
    new_tensors = {}
    embed_key = None
    for shard in sorted(set(idx["weight_map"].values())):
        with safe_open(os.path.join(SRC, shard), framework="pt") as f:
            for k in f.keys():
                if args.drop_visual and k.startswith("model.visual."):
                    continue
                if k.endswith("embed_tokens.weight"):
                    embed_key = k
                    t = f.get_tensor(k)
                    new_tensors[k] = torch.cat([t[:KEEP], t[sp_base:sp_base + n_top]], dim=0).contiguous()
                else:
                    new_tensors[k] = f.get_tensor(k)
    assert embed_key and new_tensors[embed_key].shape[0] == n_total, \
        (embed_key, new_tensors[embed_key].shape, n_total)

    # single-shard save (or shard if >40GB; our pruned model is small)
    save_file(new_tensors, f"{OUT}/model.safetensors", metadata={"format": "pt"})
    json.dump({"metadata": {"total_size": sum(t.numel() * t.element_size() for t in new_tensors.values())},
               "weight_map": {k: "model.safetensors" for k in new_tensors}},
              open(f"{OUT}/model.safetensors.index.json", "w"))

    # --- CRITICAL: remap all special token IDs in config/generation_config ---
    #     (esp. eos_token_id — if stale, generation never stops -> 60s timeout / latency blowup)
    def _remap_ids(obj):
        if isinstance(obj, int):
            return remap_sp.get(obj, obj)
        if isinstance(obj, list):
            return [_remap_ids(x) for x in obj]
        return obj
    ID_FIELDS = ["eos_token_id", "bos_token_id", "pad_token_id", "image_token_id", "video_token_id",
                 "vision_start_token_id", "vision_end_token_id", "decoder_start_token_id"]

    cfg = json.load(open(f"{SRC}/config.json"))
    cfg["text_config"]["vocab_size"] = n_total
    if "vocab_size" in cfg:
        cfg["vocab_size"] = n_total
    for scope in (cfg, cfg.get("text_config", {})):
        for fld in ID_FIELDS:
            if fld in scope:
                scope[fld] = _remap_ids(scope[fld])
    if args.drop_visual:
        cfg.pop("vision_config", None)
        for fld in ["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]:
            cfg.pop(fld, None)
    # generation_config.json too
    gcp = f"{OUT}/generation_config.json"
    if os.path.isfile(gcp):
        gc = json.load(open(gcp))
        for fld in ID_FIELDS:
            if fld in gc:
                gc[fld] = _remap_ids(gc[fld])
        json.dump(gc, open(gcp, "w"), indent=2)
    # copy remaining small json/txt configs not already written
    for fn in ["configuration.json", "LICENSE", "README.md"]:
        if os.path.isfile(f"{SRC}/{fn}"):
            shutil.copy2(f"{SRC}/{fn}", f"{OUT}/{fn}")
    json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=2)
    print(f"wrote pruned-vocab AWQ checkpoint to {OUT} (vocab {n_total}, MTP kept, drop_visual={args.drop_visual})", flush=True)


if __name__ == "__main__":
    main()
