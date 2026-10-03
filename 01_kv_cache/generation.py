"""Naive vs KV-cached generation for the tiny GPT.

At decode step t (t+1 tokens in the sequence):
    attention, cached:   O(t)      attention, naive:   O(t^2)
    MLP, cached:         O(1)      MLP, naive:         O(t)

So a full generation of T tokens costs:
    cached:   O(T^2) attention + O(T)   MLP
    naive:    O(T^3) attention + O(T^2) MLP

The benchmark in benchmark.py makes this visible as a curve.
"""
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


@torch.no_grad()
def generate_naive(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Feed the full sequence through the model at every step.

    No cache. Attention and MLP both scale with prefix length.
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Crop to block_size if needed; the model has a position limit.
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond, use_cache=False)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Prefill once, then feed one token per step with a KV cache."""
    model.eval()
    caches = None
    pos_offset = 0

    # Prefill: process the prompt in one pass, get the first next-token logits.
    logits, caches = model(idx, use_cache=True, pos_offset=0)
    pos_offset = idx.size(1)

    out = idx
    for _ in range(max_new_tokens):
        logits_last = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
            logits_last[logits_last < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits_last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=1)

        if out.size(1) >= model.cfg.block_size:
            break  # out of positions

        logits, caches = model(
            next_id, kv_caches=caches, use_cache=True, pos_offset=pos_offset
        )
        pos_offset += 1
    return out


def check_equivalence(seed=0, prompt_len=8, n_new=16):
    """Both paths must produce identical tokens under the same seed."""
    torch.manual_seed(seed)
    cfg = GPTConfig()
    model = GPT(cfg).eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len))

    torch.manual_seed(123)
    out_naive = generate_naive(model, prompt.clone(), n_new)

    torch.manual_seed(123)
    out_cached = generate_cached(model, prompt.clone(), n_new)

    assert torch.equal(out_naive, out_cached), (
        f"mismatch\nnaive : {out_naive.tolist()}\ncached: {out_cached.tolist()}"
    )
    print(f"equivalence OK ({out_naive.size(1)} tokens: {prompt_len} prompt + {n_new} new)")


if __name__ == "__main__":
    check_equivalence()
