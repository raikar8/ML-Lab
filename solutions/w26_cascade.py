"""
title: Wake-word cascade evaluation — false accepts per hour vs false rejects
week: 26
day: 2
level: medium
time: 30
tags: numpy, metrics, system-design, audio
core: yes

A two-stage detector: stage 1 (tiny, always on) scores every window; stage 2 (bigger) runs only on windows that pass
stage 1.

* `cascade_decisions(s1, s2, t1, t2)` → bool array: accepted iff `s1 ≥ t1` and `s2 ≥ t2`.
* `far_per_hour(neg_accepts, neg_hours)` and `frr(pos_accepts)` = 1 − mean(pos accepted).
* `stage2_duty(s1_neg, t1)` → fraction of negative windows that trigger stage 2 (its average compute cost).
* `pick_thresholds(pos_s1, pos_s2, neg_s1, neg_s2, neg_hours, max_fa_per_hour, grid)` — search `(t1, t2)` over
  `grid × grid`, return the pair with the **lowest FRR** subject to FA/hour ≤ target; break ties by lower stage-2 duty.
"""
import numpy as np


def cascade_decisions(s1, s2, t1, t2):
    return (np.asarray(s1) >= t1) & (np.asarray(s2) >= t2)


def far_per_hour(neg_accepts, neg_hours):
    return float(np.sum(neg_accepts) / neg_hours)


def frr(pos_accepts):
    return float(1 - np.mean(pos_accepts))


def stage2_duty(s1_neg, t1):
    return float(np.mean(np.asarray(s1_neg) >= t1))


def pick_thresholds(pos_s1, pos_s2, neg_s1, neg_s2, neg_hours, max_fa_per_hour, grid):
    best = None
    for t1 in grid:
        for t2 in grid:
            fa = far_per_hour(cascade_decisions(neg_s1, neg_s2, t1, t2), neg_hours)
            if fa > max_fa_per_hour:
                continue
            key = (frr(cascade_decisions(pos_s1, pos_s2, t1, t2)), stage2_duty(neg_s1, t1))
            if best is None or key < best[0]:
                best = (key, (float(t1), float(t2)))
    return None if best is None else best[1]
