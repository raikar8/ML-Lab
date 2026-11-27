"""
title: Metrics from scratch — precision/recall/F1, ROC-AUC, average precision, ECE
week: 2
day: 4
level: medium
time: 40
tags: numpy, metrics, calibration
core: yes

Binary labels `y` (0/1) and scores `s` (higher = more positive). NumPy only.

* `precision_recall_f1(y, yhat)` → tuple of floats (return 0.0 where a denominator is 0).
* `roc_auc(y, s)` — via ranks (Mann–Whitney U): average ranks for ties.
* `average_precision(y, s)` — sum over positives, in descending score order, of precision at that rank,
  divided by the number of positives (sklearn's definition without interpolation, ignore ties).
* `ece(y, p, n_bins=10)` — expected calibration error for predicted positive-class probabilities `p`:
  equal-width bins on [0, 1] (include 1.0 in the last bin); `sum_b (|B|/n) * |mean(y_B) − mean(p_B)|`.
"""
import numpy as np


def precision_recall_f1(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def roc_auc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s):
    y = np.asarray(y)[np.argsort(-np.asarray(s, dtype=float), kind="mergesort")]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def ece(y, p, n_bins=10):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)
