"""
title: Multi-head attention from scratch with causal and padding masks
week: 10
day: 2
level: medium
time: 30
tags: pytorch, attention, transformers, classic
core: yes

* `sdpa(q, k, v, mask=None)` — `(B, H, Tq, D)`, `(B, H, Tk, D)`, `(B, H, Tk, Dv)`; `mask` is bool broadcastable to
  `(B, H, Tq, Tk)` with **True = may attend**. Use `-inf` (or the dtype's min) for masked logits, stable softmax.
* `MultiHeadAttention(d_model, n_heads, bias=True)` with one fused `qkv` linear (`d_model → 3·d_model`) and an `out`
  linear. `forward(x, causal=False, key_padding_mask=None)` where `key_padding_mask` is bool `(B, T)` with True for real
  tokens. Output `(B, T, d_model)`.
Do not call `F.scaled_dot_product_attention` (tests compare against it).
"""
import math

import torch
import torch.nn as nn


def sdpa(q, k, v, mask=None):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, causal=False, key_padding_mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        mask = None
        if causal:
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        if key_padding_mask is not None:
            kp = key_padding_mask[:, None, None, :]
            mask = kp if mask is None else mask & kp
        o = sdpa(q, k, v, mask)
        return self.out(o.transpose(1, 2).reshape(B, T, D))
