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
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
    return lambda w, rows=slice(None): torch.clamp(torch.round(w / s[rows, 0]), -qmax - 1, qmax) * s[rows, 0]


def rtn(W, bits=4):
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
    return torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s


def gptq(W, X, bits=4, damp=0.01):
    W = W.clone().double()
    X = X.double()
    quant = make_quant(W, bits)
    H = 2 * X.T @ X
    H += damp * H.diag().mean() * torch.eye(H.shape[0], dtype=H.dtype)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    U = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    for j in range(W.shape[1]):
        w = W[:, j]
        q = quant(w)
        Q[:, j] = q
        err = (w - q) / U[j, j]
        W[:, j + 1:] -= err[:, None] * U[j, j + 1:][None]
    return Q


def output_error(W, Wq, X):
    return float(((X.double() @ W.double().T - X.double() @ Wq.double().T) ** 2).sum())
