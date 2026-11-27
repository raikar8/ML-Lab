"""
title: Multi-head Latent Attention (MLA) with latent KV cache and the absorption trick
week: 18
day: 2
level: hard
time: 50
tags: pytorch, mla, kv-cache, chinese-models
core: yes

Simplified MLA without RoPE (DeepSeek adds a small decoupled RoPE key; ignore it here).

`MLA(d, n_heads, d_head, d_latent)` with bias-free linears: `w_dkv (d → d_latent)`, `w_uk (d_latent → n_heads·d_head)`,
`w_uv (d_latent → n_heads·d_head)`, `w_q (d → n_heads·d_head)`, `w_o (n_heads·d_head → d)`.

* `forward(x, cache=None)` — **naive** path: latent `c = w_dkv(x)` (this is what's cached, shape `(B, T, d_latent)`);
  reconstruct per-head K/V from the full latent cache, causal attention for the new tokens, return `(out, latent_cache)`.
* `forward_absorbed(x, cache=None)` — same output, but never materialise K or V: fold `W_uk` into the query
  (`q̃_h = W_uk,hᵀ q_h`, scores `q̃_h · c`) and apply `W_uv` after attention over the latents (`o_h = W_uv,h (Σ p c)`).
* `cache_values_per_token(n_layers)` → number of cached scalars per token (`d_latent · n_layers`).
"""

import math

import torch

import torch.nn as nn

class MLA(nn.Module):
    def __init__(self, d, n_heads, d_head, d_latent):
        raise NotImplementedError  # TODO: implement

    def mask(self, T, Tk, device):
        raise NotImplementedError  # TODO: implement

    def forward(self, x, cache=None):
        raise NotImplementedError  # TODO: implement

    def forward_absorbed(self, x, cache=None):
        raise NotImplementedError  # TODO: implement

    def cache_values_per_token(self, n_layers):
        raise NotImplementedError  # TODO: implement
