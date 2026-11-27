"""
title: Affine quantization — scale/zero-point, symmetric vs asymmetric, SQNR, calibration
week: 21
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
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    qmax = 2 ** bits - 1
    scale = (hi - lo) / qmax if hi > lo else 1.0
    zp = int(np.clip(round(-lo / scale), 0, qmax))
    return scale, zp


def qparams_sym(amax, bits=8):
    return amax / (2 ** (bits - 1) - 1)


def quantize(x, scale, zp, qmin, qmax):
    return np.clip(np.round(x / scale) + zp, qmin, qmax).astype(np.int64)


def dequantize(q, scale, zp):
    return scale * (q.astype(float) - zp)


def sqnr_db(x, xhat):
    return float(10 * np.log10((x ** 2).sum() / ((x - xhat) ** 2).sum()))


def best_clip_mse(x, bits=8, n_grid=100):
    qm = 2 ** (bits - 1) - 1
    best, best_c = np.inf, None
    for c in np.linspace(np.abs(x).max() / n_grid, np.abs(x).max(), n_grid):
        s = c / qm
        e = ((x - dequantize(quantize(x, s, 0, -qm, qm), s, 0)) ** 2).mean()
        if e < best:
            best, best_c = e, c
    return float(best_c)
