#!/usr/bin/env python3
"""
Latency benchmark — mirrors the OFFICIAL adaptfm latency protocol exactly:
  - /invocations with {"prompt": FILLER*N, "max_tokens": M, "temperature": 0}
  - FILLER = "The quick brown fox jumps over the lazy dog. " (~repetitive on purpose)
  - 5 warmup runs, then NUM_RUNS timed; score = MEDIAN per category.
  - speedup = official_baseline_ms / median_ms   (per category, then averaged)

NOTE on hardware: official BASELINE_LATENCY was measured on A10G. We run on L4, so the
absolute ms and the printed "vs-A10G speedup" are NOT comparable to the leaderboard.
For a fair L4 number, pass --baseline <json> from our own FP16 run on the SAME GPU (GPU3).

Usage:
  CONTAINER_URL=http://localhost:8080 python bench.py --tag fp16_baseline --runs 30
  python bench.py --tag awq_ngram --runs 30 --baseline results/fp16_baseline.json
"""
import argparse, json, os, statistics, time, urllib.request

CONTAINER_URL = os.environ.get("CONTAINER_URL", "http://localhost:8080")
FILLER = "The quick brown fox jumps over the lazy dog. "
PROMPT_CONFIGS = {
    "short":  {"num_tokens": 64,   "max_new_tokens": 128},
    "medium": {"num_tokens": 2048, "max_new_tokens": 256},
    "long":   {"num_tokens": 8192, "max_new_tokens": 256},
}
A10G_BASELINE = {"short": 2582, "medium": 5441, "long": 6576}  # official, A10G — reference only


def invoke(prompt, max_tokens, timeout=600):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"{CONTAINER_URL}/invocations", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=timeout)
    out = json.loads(resp.read())
    ms = (time.perf_counter() - t0) * 1000
    text = (out.get("choices", [{}])[0].get("text", "") if out.get("choices") else "")
    return ms, len(text)


def wait_ready(timeout_s=900):
    print(f"waiting for {CONTAINER_URL}/ping ...", flush=True)
    for i in range(timeout_s // 5):
        try:
            urllib.request.urlopen(f"{CONTAINER_URL}/ping", timeout=2)
            print("ready", flush=True)
            return True
        except Exception:
            if i % 12 == 0:
                print(f"  [{i*5}s] waiting...", flush=True)
            time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--baseline", default=None, help="JSON from a same-GPU FP16 run for honest speedup")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()

    if not wait_ready():
        raise SystemExit("server never became ready")

    base = None
    if args.baseline and os.path.isfile(args.baseline):
        base = {k: v["median_ms"] for k, v in json.load(open(args.baseline))["categories"].items()}

    cats = {}
    for cat, cfg in PROMPT_CONFIGS.items():
        prompt = FILLER * max(1, cfg["num_tokens"] // 10)
        for _ in range(args.warmup):
            try: invoke(prompt, cfg["max_new_tokens"])
            except Exception: pass
        lat = []
        for i in range(args.runs):
            ms, _ = invoke(prompt, cfg["max_new_tokens"])
            lat.append(ms)
            if (i + 1) % 10 == 0:
                print(f"  [{cat}] {i+1}/{args.runs}  last={ms:.1f}ms", flush=True)
        med = round(statistics.median(lat), 2)
        cats[cat] = {"median_ms": med, "p5_ms": round(min(lat), 2),
                     "mean_ms": round(statistics.mean(lat), 2), "runs": len(lat)}
        a10 = A10G_BASELINE[cat]
        line = f"[{cat}] median={med}ms"
        if base:
            line += f"  L4-speedup={base[cat]/med:.2f}x (vs same-GPU FP16 {base[cat]}ms)"
        line += f"  | (A10G-ref {a10}ms — NOT comparable across HW)"
        print(line, flush=True)

    result = {"tag": args.tag, "url": CONTAINER_URL, "categories": cats,
              "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?")}
    if base:
        sp = {c: round(base[c] / cats[c]["median_ms"], 3) for c in cats}
        result["l4_speedup_vs_fp16"] = sp
        result["l4_avg_speedup"] = round(sum(sp.values()) / len(sp), 3)
        print(f"\n==> L4 avg speedup vs same-GPU FP16: {result['l4_avg_speedup']}x  {sp}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.tag}.json")
    json.dump(result, open(path, "w"), indent=2)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
