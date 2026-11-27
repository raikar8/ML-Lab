"""
title: Affine quantization — scale/zero-point, symmetric vs asymmetric, SQNR, calibration
week: 22
day: 1
level: easy
time: 30
tags: numpy, quantization, embedded
core: yes

* `qparams_asym(lo, hi, bits=8)` → `(scale, zero_point)` for unsigned range `[0, 2^b−1]`; first widen the range to
  include 0; zero-point is an integer.
* `qparams_sym(amax, bits=8)` → scale for signed range `[−(2^(b−1)−1), 2^(b−1)−1]`.
* `quantize(x, scale, zp, qmin, qmax)` / `dequantize(q, scale, zp)`.
* `sqnr_db(x, xhat)`.
* `best_clip_mse(x, bits=8, n_grid=100)` — symmetric clipping value `c ∈ (0, max|x|]` (grid of `n_grid` values) that
  minimises the MSE of quantize→dequantize; return `c`.
"""

import numpy as np

def qparams_asym(lo, hi, bits=8):
    raise NotImplementedError  # TODO: implement


def qparams_sym(amax, bits=8):
    raise NotImplementedError  # TODO: implement


def quantize(x, scale, zp, qmin, qmax):
    raise NotImplementedError  # TODO: implement


def dequantize(q, scale, zp):
    raise NotImplementedError  # TODO: implement


def sqnr_db(x, xhat):
    raise NotImplementedError  # TODO: implement


def best_clip_mse(x, bits=8, n_grid=100):
    raise NotImplementedError  # TODO: implement
