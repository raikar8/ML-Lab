"""
title: Wake-word cascade evaluation — false accepts per hour vs false rejects
week: 25
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
    raise NotImplementedError  # TODO: implement


def far_per_hour(neg_accepts, neg_hours):
    raise NotImplementedError  # TODO: implement


def frr(pos_accepts):
    raise NotImplementedError  # TODO: implement


def stage2_duty(s1_neg, t1):
    raise NotImplementedError  # TODO: implement


def pick_thresholds(pos_s1, pos_s2, neg_s1, neg_s2, neg_hours, max_fa_per_hour, grid):
    raise NotImplementedError  # TODO: implement
