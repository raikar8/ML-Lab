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
    T, dh = x.shape[-2], x.shape[-1]
    inv = base ** (-torch.arange(0, dh, 2, dtype=torch.float32, device=x.device) / dh)
    ang = torch.arange(T, dtype=torch.float32, device=x.device)[:, None] * inv[None]
    cos, sin = torch.cat([ang, ang], -1).cos().to(x.dtype), torch.cat([ang, ang], -1).sin().to(x.dtype)
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


class DecoderBlock(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, ffn_hidden, rope_base=10000.0):
        super().__init__()
        self.h, self.kv, self.dh, self.base = n_heads, n_kv_heads, d // n_heads, rope_base
        self.wq = nn.Linear(d, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d, n_kv_heads * self.dh, bias=False)
        self.wv = nn.Linear(d, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(n_heads * self.dh, d, bias=False)
        self.w1 = nn.Linear(d, ffn_hidden, bias=False)
        self.w3 = nn.Linear(d, ffn_hidden, bias=False)
        self.w2 = nn.Linear(ffn_hidden, d, bias=False)
        self.attn_norm = nn.RMSNorm(d, eps=1e-6)
        self.ffn_norm = nn.RMSNorm(d, eps=1e-6)

    def attn(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        q, k = rope(q, self.base), rope(k, self.base)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, -1))

    def forward(self, x):
        h = x + self.attn(self.attn_norm(x))
        z = self.ffn_norm(h)
        return h + self.w2(F.silu(self.w1(z)) * self.w3(z))


def param_count(d, n_heads, n_kv_heads, ffn_hidden):
    dh = d // n_heads
    return d * n_heads * dh * 2 + 2 * d * n_kv_heads * dh + 3 * d * ffn_hidden + 2 * d
