"""
title: Activation statistics with forward hooks (quantisation calibration style)
week: 9
day: 2
level: easy
time: 25
tags: pytorch, hooks, quantization
core: no

* `count_parameters(model, trainable_only=False)` → int (shared/tied parameters counted once).
* `ActivationRecorder(model, types=(nn.Linear, nn.Conv2d))` — a **context manager** that registers forward hooks on all
  sub-modules of the given types and records per-module running `min`, `max` and the number of calls in
  `self.stats[name] = {"min": float, "max": float, "calls": int}` (name = from `named_modules()`). All hooks must be
  removed on exit (even if an exception happened). This is exactly what a min–max quantisation observer does.
"""

import torch.nn as nn

def count_parameters(model, trainable_only=False):
    raise NotImplementedError  # TODO: implement


class ActivationRecorder:
    def __init__(self, model, types=(nn.Linear, nn.Conv2d)):
        raise NotImplementedError  # TODO: implement

    def make_hook(self, name):
        raise NotImplementedError  # TODO: implement

    def __enter__(self):
        raise NotImplementedError  # TODO: implement

    def __exit__(self, *exc):
        raise NotImplementedError  # TODO: implement
