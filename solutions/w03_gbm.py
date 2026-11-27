"""
title: Gradient boosting for regression from scratch
week: 3
day: 2
level: medium
time: 45
tags: numpy, boosting, classic
core: no

Implement `GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2)` for squared loss.

* `RegressionTree(max_depth)` with `fit(X, r)`/`predict(X)`: greedy splits minimising the sum of squared errors,
  leaves predict the mean. (Candidate thresholds: midpoints of sorted distinct values per feature; a simple
  O(n² d) search per node is fine for the test sizes, but try the O(n log n) cumulative-sum version.)
* Boosting: `F0 = mean(y)`; for each round fit a tree to the residuals `y − F`, then `F += lr * tree(X)`.
  Store the per-round training MSE in `self.train_loss_` (length `n_estimators`).
"""
import numpy as np


class RegressionTree:
    def __init__(self, max_depth=2, min_samples_leaf=2):
        self.max_depth, self.min_leaf = max_depth, min_samples_leaf

    def fit(self, X, r):
        self.root = self.grow(X, r, 0)
        return self

    def best(self, X, r):
        n = len(r)
        best = (None, None, ((r - r.mean()) ** 2).sum())
        for j in range(X.shape[1]):
            o = np.argsort(X[:, j], kind="mergesort")
            xs, rs = X[o, j], r[o]
            cs, cs2 = np.cumsum(rs), np.cumsum(rs ** 2)
            tot, tot2 = cs[-1], cs2[-1]
            for i in range(self.min_leaf - 1, n - self.min_leaf):
                if xs[i] == xs[i + 1]:
                    continue
                nl, nr = i + 1, n - i - 1
                sse = (cs2[i] - cs[i] ** 2 / nl) + ((tot2 - cs2[i]) - (tot - cs[i]) ** 2 / nr)
                if sse < best[2] - 1e-12:
                    best = (j, (xs[i] + xs[i + 1]) / 2, sse)
        return best

    def grow(self, X, r, d):
        node = {"v": float(r.mean())}
        if d >= self.max_depth or len(r) < 2 * self.min_leaf:
            return node
        j, t, _ = self.best(X, r)
        if j is None:
            return node
        m = X[:, j] <= t
        node.update(j=j, t=t, l=self.grow(X[m], r[m], d + 1), r=self.grow(X[~m], r[~m], d + 1))
        return node

    def predict(self, X):
        out = np.empty(len(X))
        for i, x in enumerate(X):
            n = self.root
            while "j" in n:
                n = n["l"] if x[n["j"]] <= n["t"] else n["r"]
            out[i] = n["v"]
        return out


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=2):
        self.n, self.lr, self.depth = n_estimators, learning_rate, max_depth

    def fit(self, X, y):
        self.f0 = float(y.mean())
        F = np.full(len(y), self.f0)
        self.trees, self.train_loss_ = [], []
        for _ in range(self.n):
            t = RegressionTree(self.depth).fit(X, y - F)
            F += self.lr * t.predict(X)
            self.trees.append(t)
            self.train_loss_.append(float(((y - F) ** 2).mean()))
        return self

    def predict(self, X):
        return self.f0 + self.lr * sum(t.predict(X) for t in self.trees)
