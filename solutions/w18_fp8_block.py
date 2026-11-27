"""
title: Simulated FP8 (E4M3) with fine-grained block scaling, DeepSeek-V3 style
week: 18
day: 3
level: medium
time: 35
tags: pytorch, fp8, quantization, chinese-models
core: no

Use PyTorch's `torch.float8_e4m3fn` dtype to simulate FP8 rounding (cast to it and back to float32). E4M3 max is 448.

* `fp8_per_tensor(x)` — one scale `s = amax/448`; return `fp8(x/s)·s`.
* `fp8_tile_1x128(x)` — activations `(M, K)` with `K % 128 == 0`: one scale per row per 128-column tile.
* `fp8_block_128(w)` — weights `(N, K)` divisible by 128: one scale per 128×128 block.
* `rel_error(x, xq)` → `‖x − xq‖ / ‖x‖`.
With a few outlier channels, the fine-grained versions must have much lower error than per-tensor.
"""
import torch

E4M3_MAX = 448.0


def fp8(x):
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def fp8_per_tensor(x):
    s = x.abs().amax().clamp_min(1e-12) / E4M3_MAX
    return fp8(x / s) * s


def fp8_tile_1x128(x):
    M, K = x.shape
    t = x.view(M, K // 128, 128)
    s = t.abs().amax(-1, keepdim=True).clamp_min(1e-12) / E4M3_MAX
    return (fp8(t / s) * s).view(M, K)


def fp8_block_128(w):
    N, K = w.shape
    b = w.view(N // 128, 128, K // 128, 128)
    s = b.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / E4M3_MAX
    return (fp8(b / s) * s).view(N, K)


def rel_error(x, xq):
    return float((x - xq).norm() / x.norm())
