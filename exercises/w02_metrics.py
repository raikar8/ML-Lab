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
    raise NotImplementedError  # TODO: implement


def rankdata(a):
    raise NotImplementedError  # TODO: implement


def roc_auc(y, s):
    raise NotImplementedError  # TODO: implement


def average_precision(y, s):
    raise NotImplementedError  # TODO: implement


def ece(y, p, n_bins=10):
    raise NotImplementedError  # TODO: implement
