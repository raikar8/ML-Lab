"""
title: SmoothQuant and Hadamard rotation for W8A8 with activation outliers
week: 23
day: 3
level: medium
time: 35
tags: pytorch, quantization, llm
core: yes

`Y = X Wᵀ` with `X (n, d)`, `W (out, d)`.

* `quant_per_row(t, bits=8)` — symmetric fake-quant with one scale per row (per token for X, per output channel for W).
* `w8a8(X, W)` → `quant_per_row(X) @ quant_per_row(W).T`.
* `smooth_scales(X, W, alpha=0.5)` → `s_j = max|X_:,j|^α / max|W_:,j|^(1−α)` (clamp small values).
* `smoothquant(X, W, alpha=0.5)` → W8A8 result using `X/s` and `W·s`.
* `hadamard(n)` — normalised Sylvester Hadamard matrix (n a power of two). `rotated(X, W)` → W8A8 of `X H` and `W H`.
"""
import torch


def quant_per_row(t, bits=8):
    qm = 2 ** (bits - 1) - 1
    s = t.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qm
    return torch.clamp(torch.round(t / s), -qm, qm) * s


def w8a8(X, W):
    return quant_per_row(X) @ quant_per_row(W).T


def smooth_scales(X, W, alpha=0.5):
    ax = X.abs().amax(0).clamp_min(1e-5)
    aw = W.abs().amax(0).clamp_min(1e-5)
    return ax ** alpha / aw ** (1 - alpha)


def smoothquant(X, W, alpha=0.5):
    s = smooth_scales(X, W, alpha)
    return w8a8(X / s, W * s)


def hadamard(n):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n ** 0.5


def rotated(X, W):
    H = hadamard(X.shape[1]).to(X.dtype)
    return w8a8(X @ H, W @ H)
