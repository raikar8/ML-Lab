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
    raise NotImplementedError  # TODO: implement


def wsd(step, total, warmup, decay_frac, peak, min_lr=0.0):
    raise NotImplementedError  # TODO: implement


def as_lambda(fn, peak, **kw):
    raise NotImplementedError  # TODO: implement
