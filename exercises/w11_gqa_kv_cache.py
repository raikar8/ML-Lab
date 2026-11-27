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
        raise NotImplementedError  # TODO: implement

    def forward(self, x, cache=None):
        raise NotImplementedError  # TODO: implement


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per=2):
    raise NotImplementedError  # TODO: implement
