"""
title: Linear algebra from scratch — matmul, Gaussian elimination, power iteration, projection
week: 0
day: 1
level: easy
time: 40
tags: linear-algebra, numpy, foundations
core: yes

Implement with plain Python loops / basic NumPy indexing (**no** `@`, `np.dot`, `np.linalg` inside your functions —
the tests compare against them):

* `matmul(A, B)` — product of two 2-D arrays; raise `ValueError` on a shape mismatch.
* `solve(A, b)` — solve the square system `Ax = b` by Gaussian elimination **with partial pivoting**
  (swap in the row with the largest |pivot|); raise `ValueError` if the matrix is singular (|pivot| < 1e-12).
  Don't modify the inputs.
* `power_iteration(A, iters=500, seed=0)` — `(eigenvalue, unit eigenvector)` for the eigenvalue of largest magnitude
  of a symmetric matrix; start from `np.random.default_rng(seed).normal(size=n)`; fix the sign so the largest-
  magnitude entry of the vector is positive. You may use your own `matmul` or `A[i] * v` sums.
* `project(A, b)` — orthogonal projection of `b` onto the column space of a full-column-rank `A`, computed by solving
  the normal equations with your `solve` (fine for well-conditioned A; note in a comment why lstsq is safer).
"""
import numpy as np


def matmul(A, B):
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"shapes {A.shape} and {B.shape} don't align")
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n))
    for i in range(m):
        for p in range(k):          # i-p-j order: the inner loop walks rows of B contiguously
            a = A[i, p]
            if a != 0.0:
                C[i, :] += a * B[p, :]
    return C


def solve(A, b):
    M = np.array(A, dtype=float)
    x = np.array(b, dtype=float).reshape(-1)
    n = M.shape[0]
    if M.shape != (n, n) or x.shape[0] != n:
        raise ValueError("A must be square and match b")
    for col in range(n):
        piv = col + int(np.argmax(np.abs(M[col:, col])))
        if abs(M[piv, col]) < 1e-12:
            raise ValueError("matrix is singular")
        if piv != col:
            M[[col, piv]] = M[[piv, col]]
            x[[col, piv]] = x[[piv, col]]
        for r in range(col + 1, n):
            f = M[r, col] / M[col, col]
            M[r, col:] -= f * M[col, col:]
            x[r] -= f * x[col]
    out = np.zeros(n)
    for r in range(n - 1, -1, -1):
        out[r] = (x[r] - (M[r, r + 1:] * out[r + 1:]).sum()) / M[r, r]
    return out


def power_iteration(A, iters: int = 500, seed: int = 0):
    A = np.asarray(A, dtype=float)
    v = np.random.default_rng(seed).normal(size=A.shape[0])
    v /= np.sqrt((v * v).sum())
    for _ in range(iters):
        w = matmul(A, v[:, None])[:, 0]
        nrm = np.sqrt((w * w).sum())
        if nrm == 0:
            return 0.0, v
        v = w / nrm
    lam = float((v * matmul(A, v[:, None])[:, 0]).sum())   # Rayleigh quotient
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return lam, v


def project(A, b):
    # Normal equations square the condition number of A; np.linalg.lstsq (QR/SVD) avoids that.
    A = np.asarray(A, dtype=float)
    AtA = matmul(A.T, A)
    Atb = matmul(A.T, np.asarray(b, dtype=float).reshape(-1, 1))[:, 0]
    return matmul(A, solve(AtA, Atb)[:, None])[:, 0]
