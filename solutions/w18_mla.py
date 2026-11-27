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
        super().__init__()
        self.h, self.dh, self.dc = n_heads, d_head, d_latent
        self.w_dkv = nn.Linear(d, d_latent, bias=False)
        self.w_uk = nn.Linear(d_latent, n_heads * d_head, bias=False)
        self.w_uv = nn.Linear(d_latent, n_heads * d_head, bias=False)
        self.w_q = nn.Linear(d, n_heads * d_head, bias=False)
        self.w_o = nn.Linear(n_heads * d_head, d, bias=False)

    def mask(self, T, Tk, device):
        return torch.ones(T, Tk, dtype=torch.bool, device=device).tril(diagonal=Tk - T)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        c = self.w_dkv(x)
        c = c if cache is None else torch.cat([cache, c], 1)
        Tk = c.shape[1]
        q = self.w_q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.w_uk(c).view(B, Tk, self.h, self.dh).transpose(1, 2)
        v = self.w_uv(c).view(B, Tk, self.h, self.dh).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        s = s.masked_fill(~self.mask(T, Tk, x.device), float("-inf"))
        o = torch.softmax(s, -1) @ v
        return self.w_o(o.transpose(1, 2).reshape(B, T, -1)), c

    def forward_absorbed(self, x, cache=None):
        B, T, _ = x.shape
        c = self.w_dkv(x)
        c = c if cache is None else torch.cat([cache, c], 1)
        Tk = c.shape[1]
        q = self.w_q(x).view(B, T, self.h, self.dh)
        Wuk = self.w_uk.weight.view(self.h, self.dh, self.dc)
        Wuv = self.w_uv.weight.view(self.h, self.dh, self.dc)
        q_lat = torch.einsum("bthd,hdc->bhtc", q, Wuk)
        s = torch.einsum("bhtc,bsc->bhts", q_lat, c) / math.sqrt(self.dh)
        s = s.masked_fill(~self.mask(T, Tk, x.device), float("-inf"))
        ctx = torch.einsum("bhts,bsc->bhtc", torch.softmax(s, -1), c)
        o = torch.einsum("bhtc,hdc->bthd", ctx, Wuv).reshape(B, T, -1)
        return self.w_o(o), c

    def cache_values_per_token(self, n_layers):
        return self.dc * n_layers
