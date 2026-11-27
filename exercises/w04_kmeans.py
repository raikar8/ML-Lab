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
    raise NotImplementedError  # TODO: implement


def kmeans_pp_init(X, k, rng):
    raise NotImplementedError  # TODO: implement


def kmeans(X, k, rng, n_iter=100, tol=1e-6):
    raise NotImplementedError  # TODO: implement
