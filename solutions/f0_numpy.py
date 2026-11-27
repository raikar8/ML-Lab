"""
title: NumPy vectorisation drills — distances, one-hot, standardise, moving average, group sums
week: 0
day: 5
level: easy
time: 35
tags: numpy, vectorisation, broadcasting, foundations
core: yes

**No Python loops** (`for`/`while`/comprehensions) in any function — the tests check the source.

* `pairwise_sq_dists(X, Y)` — `(n, m)` squared Euclidean distances via `‖x‖² + ‖y‖² − 2xᵀy`, clamped at 0.
* `one_hot(labels, k)` — `(n, k)` float array.
* `standardize(X, eps=1e-8)` — per-column zero mean, unit (population) std; constant columns become 0.
* `moving_average(x, w)` — the `len(x) − w + 1` window means using `np.cumsum` (O(n), not O(n·w)).
* `group_sum(values, groups, n_groups)` — sum of `values` per integer group id (use `np.bincount` or `np.add.at`).
* `softmax_rows(Z)` — numerically stable row-wise softmax.
"""
import numpy as np


def pairwise_sq_dists(X, Y):
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    d2 = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2.0 * X @ Y.T
    return np.maximum(d2, 0.0)


def one_hot(labels, k: int):
    return np.eye(k)[np.asarray(labels)]


def standardize(X, eps: float = 1e-8):
    X = np.asarray(X, dtype=float)
    mu = X.mean(0)
    sd = X.std(0)
    return (X - mu) / np.where(sd < eps, 1.0, sd)


def moving_average(x, w: int):
    c = np.cumsum(np.concatenate([[0.0], np.asarray(x, dtype=float)]))
    return (c[w:] - c[:-w]) / w


def group_sum(values, groups, n_groups: int):
    return np.bincount(np.asarray(groups), weights=np.asarray(values, dtype=float), minlength=n_groups)


def softmax_rows(Z):
    Z = np.asarray(Z, dtype=float)
    e = np.exp(Z - Z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
