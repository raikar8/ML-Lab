"""
title: k-means with k-means++ initialisation
week: 4
day: 1
level: medium
time: 30
tags: numpy, clustering, classic
core: yes

* `kmeans_pp_init(X, k, rng)` → `(k, d)` initial centres: first uniform at random, then each next centre sampled with
  probability ∝ squared distance to the nearest chosen centre (`rng` is a `np.random.Generator`).
* `kmeans(X, k, rng, n_iter=100, tol=1e-6)` → `(centers, labels, inertia)`. Vectorised assignment step; if a cluster
  becomes empty, re-seed it with the point farthest from its current centre. Stop when centres move less than `tol`.
"""
import numpy as np


def sq_dists(X, C):
    return np.maximum((X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None], 0.0)


def kmeans_pp_init(X, k, rng):
    n = len(X)
    centers = [X[rng.integers(n)]]
    d2 = sq_dists(X, np.array(centers))[:, 0]
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        c = X[rng.choice(n, p=p)]
        centers.append(c)
        d2 = np.minimum(d2, ((X - c) ** 2).sum(1))
    return np.array(centers)


def kmeans(X, k, rng, n_iter=100, tol=1e-6):
    C = kmeans_pp_init(X, k, rng)
    for _ in range(n_iter):
        D = sq_dists(X, C)
        labels = D.argmin(1)
        newC = C.copy()
        for j in range(k):
            pts = X[labels == j]
            if len(pts):
                newC[j] = pts.mean(0)
            else:
                newC[j] = X[D[np.arange(len(X)), labels].argmax()]
        shift = np.abs(newC - C).max()
        C = newC
        if shift < tol:
            break
    D = sq_dists(X, C)
    labels = D.argmin(1)
    return C, labels, float(D[np.arange(len(X)), labels].sum())
