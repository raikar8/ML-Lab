"""
title: GPTQ — Hessian-aware column-by-column weight quantization
week: 22
day: 2
level: hard
time: 50
tags: pytorch, quantization, llm
core: yes

Weights `W (out, in)`, calibration inputs `X (n, in)`. Quantizer: symmetric 4-bit, **per output row** scale
`s = max|W_row| / 7` computed once from the original W (keep it fixed), values in [−8, 7].

* `rtn(W, bits=4)` — round-to-nearest baseline with the same scales.
* `gptq(W, X, bits=4, damp=0.01)` — `H = 2 XᵀX + damp·mean(diag)·I`; `Hinv = cholesky_inverse` → take the **upper**
  Cholesky factor `U` of `H⁻¹`. For each column j: `q = quant(w_j)`, `err = (w_j − q) / U[j, j]`,
  `W[:, j+1:] −= err ⊗ U[j, j+1:]`. Return the quantized (dequantized) matrix.
* `output_error(W, Wq, X)` → `‖XWᵀ − XWqᵀ‖² `.
GPTQ must beat RTN on correlated inputs.
"""

import torch

def make_quant(W, bits):
    raise NotImplementedError  # TODO: implement


def rtn(W, bits=4):
    raise NotImplementedError  # TODO: implement


def gptq(W, X, bits=4, damp=0.01):
    raise NotImplementedError  # TODO: implement


def output_error(W, Wq, X):
    raise NotImplementedError  # TODO: implement
