"""
title: Q15 fixed-point arithmetic and a fixed-point FIR filter
week: 20
day: 3
level: medium
time: 35
tags: python, fixed-point, dsp, embedded
core: yes

Work with Python ints (or int64 NumPy) as if they were int16/int32 registers.

* `to_q15(x)` — float in [−1, 1) → int with rounding and saturation to [−32768, 32767]. `from_q15(q)` → float.
* `q15_mul(a, b)` — Q15 × Q15 → Q15: 32-bit product, add rounding constant `1 << 14`, arithmetic shift right 15, saturate.
* `q15_add(a, b)` — saturating add.
* `fir_q15(x_q15, h_q15)` — causal FIR `y[n] = Σ_k h[k]·x[n−k]` with products accumulated in a wide (Q30) accumulator,
  then rounded and shifted back to Q15 once per output sample, saturated. Inputs are NumPy int arrays; return int64 array.
"""

import numpy as np

QMAX, QMIN = 32767, -32768

def sat16(v):
    raise NotImplementedError  # TODO: implement


def to_q15(x):
    raise NotImplementedError  # TODO: implement


def from_q15(q):
    raise NotImplementedError  # TODO: implement


def q15_mul(a, b):
    raise NotImplementedError  # TODO: implement


def q15_add(a, b):
    raise NotImplementedError  # TODO: implement


def fir_q15(x_q15, h_q15):
    raise NotImplementedError  # TODO: implement
