"""
title: Vectorised k-nearest neighbours
week: 3
day: 4
level: easy
time: 20
tags: numpy, knn, vectorisation
core: yes

* `pairwise_sq_dists(A, B)` → `(n, m)` matrix of squared Euclidean distances using
  `|a|² + |b|² − 2 a·b` (one matmul, **no Python loops**); clip tiny negatives to 0.
* `knn_predict(X_train, y_train, X_test, k)` → majority label among the k nearest (ties → smallest label).
  Use `np.argpartition` for O(n) selection per row.
"""

import numpy as np

def pairwise_sq_dists(A, B):
    raise NotImplementedError  # TODO: implement


def knn_predict(X_train, y_train, X_test, k):
    raise NotImplementedError  # TODO: implement
