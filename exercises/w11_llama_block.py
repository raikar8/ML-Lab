"""
title: A modern decoder block — RMSNorm, RoPE, GQA, SwiGLU (no biases)
week: 11
day: 1
level: hard
time: 50
tags: pytorch, transformers, architecture
core: yes

Implement `DecoderBlock(d, n_heads, n_kv_heads, ffn_hidden, rope_base=10000.0)`:

```
h = x + Attn(RMSNorm(x));  y = h + FFN(RMSNorm(h))
```

* Attention: `wq (d → n_heads·dh)`, `wk`, `wv (d → n_kv·dh)`, `wo (n_heads·dh → d)`, all `bias=False`, `dh = d // n_heads`;
  RoPE (half-split) on q and k; **GQA**: repeat each KV head `n_heads // n_kv` times; causal attention (you may use
  `F.scaled_dot_product_attention` here).
* FFN: SwiGLU `w2(silu(w1 x) * w3 x)` with `bias=False`.
* Norms: `nn.RMSNorm(d, eps=1e-6)` attributes `attn_norm`, `ffn_norm`.
* `param_count(d, n_heads, n_kv_heads, ffn_hidden)` — the block's parameter count from the formula (no module).
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

def rope(x, base):
    raise NotImplementedError  # TODO: implement


class DecoderBlock(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, ffn_hidden, rope_base=10000.0):
        raise NotImplementedError  # TODO: implement

    def attn(self, x):
        raise NotImplementedError  # TODO: implement

    def forward(self, x):
        raise NotImplementedError  # TODO: implement


def param_count(d, n_heads, n_kv_heads, ffn_hidden):
    raise NotImplementedError  # TODO: implement
