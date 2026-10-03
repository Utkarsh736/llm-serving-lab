"""Tiny GPT for LLM Serving Lab, Milestone 1.

Design notes:
- Learned positional embeddings (simpler than RoPE; we're studying
  inference mechanics, not training quality).
- Attention accepts an optional (past_k, past_v) pair so the same
  forward pass supports both naive and cached generation.
- Random vocab (no tokenizer) because we care about inference cost,
  not text quality.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 4096
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if use_cache:
            if kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        Tk = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask only needed when q has more than one position
        # (prefill). For single-token decode, every key is visible.
        if T > 1:
            q_pos = torch.arange(Tk - T, Tk, device=x.device)
            k_pos = torch.arange(Tk, device=x.device)
            mask = q_pos[:, None] >= k_pos[None, :]
            att = att.masked_fill(~mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), new_cache


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return F.dropout(x, p=self.dropout, training=self.training)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None, use_cache=False):
        h, new_cache = self.attn(self.ln_1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, kv_caches=None, use_cache=False, pos_offset=0):
        """idx: (B, T) token ids.

        kv_caches: list of per-layer (k, v) tuples, or None on prefill.
        pos_offset: starting position for positional embeddings. For
                    prefill it's 0; for decode step t it's t.
        """
        B, T = idx.shape
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=past, use_cache=use_cache)
            if use_cache:
                new_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_caches


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfig()
    model = GPT(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits, _ = model(x)
    print(f"input: {tuple(x.shape)}  logits: {tuple(logits.shape)}")
    assert logits.shape == (2, 16, cfg.vocab_size)
    print("smoke test OK")
