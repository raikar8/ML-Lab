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
    raise NotImplementedError  # TODO: implement


def one_hot(labels, k: int):
    raise NotImplementedError  # TODO: implement


def standardize(X, eps: float = 1e-8):
    raise NotImplementedError  # TODO: implement


def moving_average(x, w: int):
    raise NotImplementedError  # TODO: implement


def group_sum(values, groups, n_groups: int):
    raise NotImplementedError  # TODO: implement


def softmax_rows(Z):
    raise NotImplementedError  # TODO: implement
