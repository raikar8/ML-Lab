"""
title: Integer-only int8 linear layer with fixed-point requantization
week: 21
day: 3
level: hard
time: 45
tags: numpy, quantization, embedded, classic
core: yes

Weights: symmetric per-output-channel int8 (`zp = 0`); activations: asymmetric int8 in `[−128, 127]` with zero-point.

* `quantize_multiplier(M)` — for real `0 < M < 1`, return `(M0, shift)` with `M ≈ M0 · 2^(−31) · 2^(−shift)`,
  `M0 ∈ [2^30, 2^31)` an int (Q31 in [0.5, 1)).
* `requantize(acc, M0, shift, zp_out)` — `round(acc · M0 / 2^31)` then rounding right shift by `shift`, add `zp_out`, clamp to
  int8. Use Python ints / int64 with explicit rounding (`(v + (1 << (n−1))) >> n`), no floats.
* `int8_linear(q_x, zp_x, q_w, b_q, M0s, shifts, zp_y)` — `q_x (N, K)` int8, `q_w (C, K)` int8, `b_q (C,)` int32 bias with
  scale `s_x·s_w[c]`; `acc = Σ (q_x − zp_x)·q_w + b_q` in int32/int64; per-channel requantize. Return int8 `(N, C)`.

The test builds a float layer, quantizes it, and checks the integer output is within 1 LSB of the float result.
"""
import math

import numpy as np


def quantize_multiplier(M):
    m, e = math.frexp(M)          # M = m * 2**e, m in [0.5, 1)
    M0 = round(m * (1 << 31))
    if M0 == 1 << 31:
        M0 //= 2
        e += 1
    return int(M0), int(-e)


def rshift_round(v, n):
    return (v + (1 << (n - 1))) >> n if n > 0 else v


def requantize(acc, M0, shift, zp_out):
    acc = np.asarray(acc, dtype=object)
    v = np.vectorize(lambda a: rshift_round(rshift_round(int(a) * M0, 31), shift) + zp_out, otypes=[object])(acc)
    return np.clip(v.astype(np.int64), -128, 127)


def int8_linear(q_x, zp_x, q_w, b_q, M0s, shifts, zp_y):
    acc = (q_x.astype(np.int64) - zp_x) @ q_w.astype(np.int64).T + b_q.astype(np.int64)[None]
    out = np.empty_like(acc)
    for c in range(acc.shape[1]):
        out[:, c] = requantize(acc[:, c], M0s[c], shifts[c], zp_y)
    return out
