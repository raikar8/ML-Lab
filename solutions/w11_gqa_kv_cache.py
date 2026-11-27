"""
title: GQA attention with a KV cache — incremental decoding equals the full pass
week: 11
day: 2
level: hard
time: 45
tags: pytorch, kv-cache, inference, transformers
core: yes

`GQAttention(d, n_heads, n_kv_heads)` (bias-free `wq, wk, wv, wo`, no RoPE for simplicity):

* `forward(x, cache=None)` — `x (B, T, d)`. If `cache` is `None`, run causal attention over `x` and return
  `(out, new_cache)`. If `cache = (k, v)` with shapes `(B, n_kv, T_past, dh)`, the new tokens attend to all cached
  positions plus causally to themselves; append the new K/V to the cache. **Cache K/V before repeating for GQA**
  (that's the memory saving).
* `kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per=2)` → int.

Test: prefill on the first part of a sequence, then feed the remaining tokens one by one; outputs must equal a full
forward pass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAttention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads):
        super().__init__()
        self.h, self.kv, self.dh = n_heads, n_kv_heads, d // n_heads
        self.wq = nn.Linear(d, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d, n_kv_heads * self.dh, bias=False)
        self.wv = nn.Linear(d, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(n_heads * self.dh, d, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        if cache is not None:
            k = torch.cat([cache[0], k], 2)
            v = torch.cat([cache[1], v], 2)
        new_cache = (k, v)
        Tk = k.shape[2]
        past = Tk - T
        mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=past)
        rep = self.h // self.kv
        o = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), attn_mask=mask)
        return self.wo(o.transpose(1, 2).reshape(B, T, -1)), new_cache


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per
