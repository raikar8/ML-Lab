"""
title: Decision-tree split finder and a small CART classifier
week: 3
day: 1
level: medium
time: 45
tags: numpy, trees, classic
core: no

1. `gini(counts)` — Gini impurity from a 1-D array of class counts.
2. `best_split(x, y, n_classes)` — for one numeric feature `x` and integer labels `y`, return
   `(threshold, weighted_child_gini)` minimising the weighted Gini of `x <= t` vs `x > t`. Sort once and sweep with
   incremental counts (O(n log n)); thresholds are midpoints between consecutive **distinct** values. Return
   `(None, inf)` if there's no valid split.
3. `DecisionTree(max_depth, min_samples_leaf=1)` with `fit(X, y)` and `predict(X)` (majority class per leaf),
   choosing the best feature and threshold at each node.
"""
import numpy as np


def gini(counts):
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return float(1.0 - (p * p).sum())


def best_split(x, y, n_classes):
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    n = len(xs)
    left = np.zeros(n_classes)
    right = np.bincount(ys, minlength=n_classes).astype(float)
    best_t, best = None, np.inf
    for i in range(n - 1):
        left[ys[i]] += 1
        right[ys[i]] -= 1
        if xs[i] == xs[i + 1]:
            continue
        nl, nr = i + 1, n - i - 1
        score = (nl * gini(left) + nr * gini(right)) / n
        if score < best:
            best, best_t = score, (xs[i] + xs[i + 1]) / 2
    return best_t, best


class DecisionTree:
    def __init__(self, max_depth=3, min_samples_leaf=1):
        self.max_depth, self.min_leaf = max_depth, min_samples_leaf
        self.root = None

    def fit(self, X, y):
        self.n_classes = int(y.max()) + 1
        self.root = self.grow(X, y, 0)
        return self

    def grow(self, X, y, depth):
        counts = np.bincount(y, minlength=self.n_classes)
        node = {"pred": int(counts.argmax())}
        if depth >= self.max_depth or gini(counts) == 0.0 or len(y) < 2 * self.min_leaf:
            return node
        best = (None, None, gini(counts))
        for j in range(X.shape[1]):
            t, sc = best_split(X[:, j], y, self.n_classes)
            if t is not None and sc < best[2]:
                m = X[:, j] <= t
                if m.sum() >= self.min_leaf and (~m).sum() >= self.min_leaf:
                    best = (j, t, sc)
        j, t, _ = best
        if j is None:
            return node
        m = X[:, j] <= t
        node.update(feat=j, thr=t, left=self.grow(X[m], y[m], depth + 1), right=self.grow(X[~m], y[~m], depth + 1))
        return node

    def predict(self, X):
        out = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            n = self.root
            while "feat" in n:
                n = n["left"] if x[n["feat"]] <= n["thr"] else n["right"]
            out[i] = n["pred"]
        return out
