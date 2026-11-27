"""
title: Rotary position embeddings (RoPE) — and prove the relative property
week: 10
day: 5
level: medium
time: 30
tags: pytorch, positional, transformers
core: yes

Use the **half-split** convention (Llama / Hugging Face `rotate_half`): dimension `i < d/2` is paired with `i + d/2`.

* `rope_cache(T, d, base=10000.0, offset=0)` → `(cos, sin)` each `(T, d)` for positions `offset … offset+T-1`, with
  `inv_freq[i] = base^(-2i/d)` for `i < d/2`, duplicated for both halves.
* `rotate_half(x)` → `cat(-x2, x1)` on the last dim.
* `apply_rope(x, cos, sin)` → `x*cos + rotate_half(x)*sin`; `x` is `(…, T, d)`.
* `partial_rope(x, cos, sin, fraction)` — rotate only the first `int(d*fraction)` dims (cos/sin built for that size),
  leave the rest unchanged (as in Qwen3.5's 25% partial RoPE).
"""
import torch


def rope_cache(T, d, base=10000.0, offset=0):
    inv = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    pos = torch.arange(offset, offset + T, dtype=torch.float64)
    ang = pos[:, None] * inv[None]
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], -1)


def apply_rope(x, cos, sin):
    return x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)


def partial_rope(x, cos, sin, fraction):
    r = int(x.shape[-1] * fraction)
    return torch.cat([apply_rope(x[..., :r], cos, sin), x[..., r:]], -1)
