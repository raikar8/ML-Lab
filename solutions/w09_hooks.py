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
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


class ActivationRecorder:
    def __init__(self, model, types=(nn.Linear, nn.Conv2d)):
        self.model, self.types = model, types
        self.stats, self.handles = {}, []

    def make_hook(self, name):
        def hook(mod, inp, out):
            s = self.stats.setdefault(name, {"min": float("inf"), "max": float("-inf"), "calls": 0})
            s["min"] = min(s["min"], float(out.detach().min()))
            s["max"] = max(s["max"], float(out.detach().max()))
            s["calls"] += 1
        return hook

    def __enter__(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, self.types):
                self.handles.append(mod.register_forward_hook(self.make_hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False
