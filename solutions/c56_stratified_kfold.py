"""
title: Stratified k-fold cross-validation from scratch
pattern: ML from scratch
level: medium
time: 30
frequency: high
week: 2
day: 5
tags: numpy, evaluation, cross-validation
complexity: O(n log n) (grouping by label) time, O(n) space
hint: Shuffle the indices of each class separately, then deal them out to folds round-robin.
hint: Keep a running "next fold" pointer across classes so fold sizes differ by at most one.
follow-up: Group k-fold: all samples of the same speaker must stay in one fold. Why does this matter for audio?
follow-up: Time-series split: why is random k-fold wrong for forecasting, and what do you do instead?

Write `stratified_kfold(y, k, seed=0)` returning a list of `k` pairs `(train_idx, val_idx)` (sorted NumPy int
arrays) such that

* every index appears in exactly one validation fold, and train = all other indices;
* for every class, its count in each validation fold differs by at most 1 across folds;
* the total size of the validation folds differs by at most 1 across folds;
* results are reproducible for a given `seed` (use `np.random.default_rng(seed)`).

Also write `cross_val_score(model_fn, X, y, k, seed=0)` that, for each fold, calls
`model_fn(X_train, y_train)` → a `predict(X)` callable, and returns the list of validation accuracies.
Raise `ValueError` if `k < 2` or `k > len(y)`.
"""
import numpy as np


def stratified_kfold(y, k: int, seed: int = 0):
    y = np.asarray(y)
    n = len(y)
    if k < 2 or k > n:
        raise ValueError("need 2 <= k <= len(y)")
    rng = np.random.default_rng(seed)
    fold_of = np.empty(n, dtype=int)
    nxt = 0
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold_of[idx] = (nxt + np.arange(len(idx))) % k
        nxt = (nxt + len(idx)) % k
    out = []
    for f in range(k):
        val = np.flatnonzero(fold_of == f)
        train = np.flatnonzero(fold_of != f)
        out.append((train, val))
    return out


def cross_val_score(model_fn, X, y, k: int, seed: int = 0) -> list[float]:
    X, y = np.asarray(X), np.asarray(y)
    scores = []
    for tr, va in stratified_kfold(y, k, seed):
        predict = model_fn(X[tr], y[tr])
        scores.append(float(np.mean(np.asarray(predict(X[va])) == y[va])))
    return scores
