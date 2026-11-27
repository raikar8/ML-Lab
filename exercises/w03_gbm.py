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
        raise NotImplementedError  # TODO: implement

    def fit(self, X, r):
        raise NotImplementedError  # TODO: implement

    def best(self, X, r):
        raise NotImplementedError  # TODO: implement

    def grow(self, X, r, d):
        raise NotImplementedError  # TODO: implement

    def predict(self, X):
        raise NotImplementedError  # TODO: implement


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=2):
        raise NotImplementedError  # TODO: implement

    def fit(self, X, y):
        raise NotImplementedError  # TODO: implement

    def predict(self, X):
        raise NotImplementedError  # TODO: implement
