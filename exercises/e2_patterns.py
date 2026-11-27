"""
title: Design patterns for training code — registry/factory, strategy, observer callbacks
week: 0
day: 4
level: medium
time: 45
tags: oop, design-patterns, solid, training, elective
core: yes

* `Registry(kind)` — `register(name)` is a **decorator** that stores a class or function under `name` (raise `KeyError`
  on a duplicate name) and returns it unchanged; `build(name, **kwargs)` calls the stored object with kwargs (raise
  `KeyError` listing the available names for an unknown one); `names()` is a sorted list; `in` works.
* `default_losses()` — return a fresh `Registry("loss")` with two **strategies** registered through the decorator:
  `"mse"` → `MSE` and `"huber"` → `Huber(delta=1.0)`, both callables `loss(pred, target) -> float` on NumPy arrays (mean over elements).
* `Trainer(step_fn, callbacks=())` — `fit(epochs)` calls `step_fn(epoch) -> float` (the epoch's validation loss), then
  every callback's `on_epoch_end(trainer, epoch, loss)`; stops early when a callback sets `trainer.stop = True`.
  Records `history` (list of losses). The trainer knows nothing about specific callbacks (**open/closed**).
* Callbacks: `EarlyStopping(patience, min_delta=0.0)` — stop after `patience` epochs without an improvement of more
  than `min_delta` over the best loss so far; stores `best` and `best_epoch`. `History()` — records `(epoch, loss)`
  pairs in `records`.
"""

import numpy as np

class Registry:
    def __init__(self, kind: str):
        raise NotImplementedError  # TODO: implement

    def register(self, name: str):
        raise NotImplementedError  # TODO: implement

    def build(self, name: str, **kwargs):
        raise NotImplementedError  # TODO: implement

    def names(self) -> list[str]:
        raise NotImplementedError  # TODO: implement

    def __contains__(self, name) -> bool:
        raise NotImplementedError  # TODO: implement


class MSE:
    def __call__(self, pred, target) -> float:
        raise NotImplementedError  # TODO: implement


class Huber:
    def __init__(self, delta: float = 1.0):
        raise NotImplementedError  # TODO: implement

    def __call__(self, pred, target) -> float:
        raise NotImplementedError  # TODO: implement


def default_losses() -> Registry:
    raise NotImplementedError  # TODO: implement


class Trainer:
    def __init__(self, step_fn, callbacks=()):
        raise NotImplementedError  # TODO: implement

    def fit(self, epochs: int):
        raise NotImplementedError  # TODO: implement


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        raise NotImplementedError  # TODO: implement

    def on_epoch_end(self, trainer, epoch, loss):
        raise NotImplementedError  # TODO: implement


class History:
    def __init__(self):
        raise NotImplementedError  # TODO: implement

    def on_epoch_end(self, trainer, epoch, loss):
        raise NotImplementedError  # TODO: implement
