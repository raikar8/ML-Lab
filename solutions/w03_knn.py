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
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return np.maximum(d, 0.0)


def knn_predict(X_train, y_train, X_test, k):
    d = pairwise_sq_dists(X_test, X_train)
    idx = np.argpartition(d, kth=min(k, d.shape[1]) - 1, axis=1)[:, :k]
    labels = y_train[idx]
    n_cls = int(y_train.max()) + 1
    counts = np.apply_along_axis(lambda r: np.bincount(r, minlength=n_cls), 1, labels)
    return counts.argmax(1)
