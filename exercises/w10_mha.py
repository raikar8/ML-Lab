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
    raise NotImplementedError  # TODO: implement


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        raise NotImplementedError  # TODO: implement

    def forward(self, x, causal=False, key_padding_mask=None):
        raise NotImplementedError  # TODO: implement
