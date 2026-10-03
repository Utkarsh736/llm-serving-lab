"""Benchmark naive vs KV-cached generation.

We measure per-step cost as a function of sequence length, rather than
running full generations. Two reasons:

1. A full naive generation at L=2048 on CPU takes ~30 minutes; the shape
   of the curve is already clear at L=128.
2. The cost of a naive decode step at position L is identical to the cost
   of prefilling a length-L prompt. We measure once, report both.

Outputs:
  results.csv      raw timings
  benchmark.png    decode step time vs sequence length (log-log)
"""
from __future__ import annotations
import csv
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; no GUI needed
import matplotlib.pyplot as plt
import torch

from model import GPT, GPTConfig


SEQ_LENGTHS = [128, 512, 2048]
N_RUNS = 3
N_WARMUP = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _time(fn, n_runs=N_RUNS, n_warmup=N_WARMUP):
    """Median wall time of fn() over n_runs, after n_warmup discarded runs.

    Median, not mean: a stray GC pause or OS scheduler blip shouldn't
    distort the result. Warmup matters because PyTorch allocates lazily
    and the first call also warms caches.
    """
    for _ in range(n_warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench_forward(model, L):
    """Time one forward pass over L tokens. This is prefill(L) AND naive decode at L."""
    x = torch.randint(0, model.cfg.vocab_size, (1, L), device=DEVICE)
    with torch.no_grad():
        return _time(lambda: model(x, use_cache=False))


def bench_cached_decode(model, L):
    """Time one single-token forward with a cache of size L."""
    prompt = torch.randint(0, model.cfg.vocab_size, (1, L), device=DEVICE)
    with torch.no_grad():
        _, caches = model(prompt, use_cache=True, pos_offset=0)

    new_tok = torch.randint(0, model.cfg.vocab_size, (1, 1), device=DEVICE)

    def step():
        model(new_tok, kv_caches=caches, use_cache=True, pos_offset=L)

    with torch.no_grad():
        return _time(step)


def main():
    torch.manual_seed(0)
    cfg = GPTConfig()
    model = GPT(cfg).to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE}")
    print(f"model:  {n_params/1e6:.2f}M params\n")

    rows = []
    for L in SEQ_LENGTHS:
        print(f"seq_len = {L}")
        t_prefill = bench_forward(model, L)
        t_naive   = t_prefill  # same computation; aliased for clarity in output
        t_cached  = bench_cached_decode(model, L)

        speedup = t_naive / t_cached
        print(f"  prefill        : {t_prefill*1000:8.2f} ms")
        print(f"  decode naive   : {t_naive*1000:8.2f} ms  (same computation)")
        print(f"  decode cached  : {t_cached*1000:8.2f} ms")
        print(f"  speedup        : {speedup:8.1f}x\n")

        rows.append({
            "seq_len": L,
            "prefill_ms": t_prefill * 1000,
            "decode_naive_ms": t_naive * 1000,
            "decode_cached_ms": t_cached * 1000,
            "speedup": speedup,
        })

    out_dir = Path(__file__).parent

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [r["seq_len"] for r in rows]
    ax.plot(xs, [r["decode_naive_ms"] for r in rows], "o-", label="naive decode")
    ax.plot(xs, [r["decode_cached_ms"] for r in rows], "o-", label="cached decode")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("time per decode step (ms)")
    ax.set_title(f"Decode step cost vs sequence length ({DEVICE})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = out_dir / "benchmark.png"
    fig.savefig(png_path, dpi=120)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
