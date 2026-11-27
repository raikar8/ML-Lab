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
    raise NotImplementedError  # TODO: implement


def best_split(x, y, n_classes):
    raise NotImplementedError  # TODO: implement


class DecisionTree:
    def __init__(self, max_depth=3, min_samples_leaf=1):
        raise NotImplementedError  # TODO: implement

    def fit(self, X, y):
        raise NotImplementedError  # TODO: implement

    def grow(self, X, y, depth):
        raise NotImplementedError  # TODO: implement

    def predict(self, X):
        raise NotImplementedError  # TODO: implement
