# LLM Serving Lab

A pedagogical reimplementation of the LLM serving stack — KV cache,
continuous batching, paged memory, and a gateway — written to understand
how vLLM, TGI, and TensorRT-LLM actually work.

Five layers, built in order:

| Layer | Directory | Status |
|---|---|---|
| KV cache | `01_kv_cache/` | ✅ done |
| Continuous batching | `02_batching_sim/` | planned |
| Paged memory | `03_paged_memory/` | planned |
| vLLM config sweep | `04_serving_benchmarks/` | planned |
| FastAPI gateway | `05_gateway/` | planned |

## 1. KV cache

### What it is

Autoregressive generation predicts one token at a time: each step feeds the
whole sequence-so-far through the model and samples the next token. The naive
implementation recomputes attention over the entire prefix at every step,
which makes each decode step O(L²) in attention and O(L) in MLP, where L is
the current sequence length.

A KV cache stores the key and value tensors from every layer after each step
and reuses them. The decode step then computes attention for only the new
token — O(L) instead of O(L²) — and runs the MLP for one token — O(1) instead
of O(L).

### What we built

- A tiny GPT (~13.8M params, 6 layers, 6 heads, 384 dim) with
  KV-cache-aware attention (`01_kv_cache/model.py`).
- Two generation paths: `generate_naive` (recompute every step) and
  `generate_cached` (prefill once, then one token per step).
- An equivalence test proving both paths produce bit-identical output.
- A benchmark harness that isolates the cost of a single decode step at
  sequence lengths 128, 512, and 2048.

### Results (CPU, fp32)

| Sequence length | Naive decode | Cached decode | Speedup |
|---|---|---|---|
| 128  | 19.24 ms  | 4.89 ms  | 3.9×  |
| 512  | 105.94 ms | 7.05 ms  | 15.0× |
| 2048 | 1082.12 ms| 17.02 ms | 63.6× |

![Decode cost vs sequence length](01_kv_cache/benchmark.png)

### What the numbers mean

**Speedup grows with sequence length.** The naive path's cost per decode step
scales quadratically with the prefix; the cached path scales linearly. Over a
full generation of T tokens, this is the difference between O(T³) and O(T²)
total attention work — which is why no production LLM serves without a
KV cache.

**Cached decode has a constant floor.** Fitting `T(L) = a + b·L` to the
cached measurements gives a ≈ 4.1 ms (the per-step constant — MLP, LayerNorms,
embeddings, sampling) and b ≈ 0.0063 ms per cached token (the linear attention
term). At L=128 the floor is ~83% of the cost; at L=2048 it's ~24%. This floor
is the target of the next layer: **continuous batching exists to amortize the
constant per-step cost across concurrent requests.**

**The cache itself costs memory.** KV cache size per sequence is
`2 × n_layer × n_head × head_dim × L × dtype_bytes`. At L=2048, fp32, that's
≈ 37.7 MB per sequence — linear in both sequence length and concurrency. This
is what Milestone 3's paged memory model will manage.

### Reproduce

```bash
uv run 01_kv_cache/benchmark.py
```
