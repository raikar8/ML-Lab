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
    raise NotImplementedError  # TODO: implement


def solve(A, b):
    raise NotImplementedError  # TODO: implement


def power_iteration(A, iters: int = 500, seed: int = 0):
    raise NotImplementedError  # TODO: implement


def project(A, b):
    # Normal equations square the condition number of A; np.linalg.lstsq (QR/SVD) avoids that.
    raise NotImplementedError  # TODO: implement
