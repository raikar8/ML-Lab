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
    raise NotImplementedError  # TODO: implement


def rshift_round(v, n):
    raise NotImplementedError  # TODO: implement


def requantize(acc, M0, shift, zp_out):
    raise NotImplementedError  # TODO: implement


def int8_linear(q_x, zp_x, q_w, b_q, M0s, shifts, zp_y):
    raise NotImplementedError  # TODO: implement
