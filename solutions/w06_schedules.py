"""
title: Learning-rate schedules — warmup+cosine and WSD
week: 6
day: 2
level: easy
time: 20
tags: pytorch, schedules
core: no

Pure functions of the step (0-indexed) returning the learning rate:

* `warmup_cosine(step, total, warmup, peak, min_lr=0.0)` — linear warmup from `peak/warmup` at step 0 to `peak` at
  step `warmup-1`, then cosine decay from `peak` to `min_lr` over the remaining steps (`min_lr` at step `total-1`).
* `wsd(step, total, warmup, decay_frac, peak, min_lr=0.0)` — warmup (as above), stable at `peak`, then **linear**
  decay to `min_lr` over the last `round(decay_frac*total)` steps (reaching `min_lr` at `total-1`).
* `as_lambda(fn, peak, **kw)` — a `LambdaLR`-compatible factor function `step -> fn(step, peak=peak, **kw)/peak`.
"""
import math


def warmup_cosine(step, total, warmup, peak, min_lr=0.0):
    if step < warmup:
        return peak * (step + 1) / warmup
    span = max(1, total - 1 - (warmup - 1))
    t = min(1.0, (step - (warmup - 1)) / span)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * t))


def wsd(step, total, warmup, decay_frac, peak, min_lr=0.0):
    if step < warmup:
        return peak * (step + 1) / warmup
    n_decay = round(decay_frac * total)
    start = total - n_decay
    if step < start or n_decay == 0:
        return peak
    t = (step - start + 1) / n_decay
    return peak + (min_lr - peak) * min(1.0, t)


def as_lambda(fn, peak, **kw):
    return lambda step: fn(step, peak=peak, **kw) / peak
