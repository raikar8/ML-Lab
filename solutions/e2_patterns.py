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
        self.kind = kind
        self._items: dict = {}

    def register(self, name: str):
        def deco(obj):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj
        return deco

    def build(self, name: str, **kwargs):
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'; available: {', '.join(self.names())}")
        return self._items[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name) -> bool:
        return name in self._items


class MSE:
    def __call__(self, pred, target) -> float:
        d = np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)
        return float(np.mean(d * d))


class Huber:
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, pred, target) -> float:
        a = np.abs(np.asarray(pred, dtype=float) - np.asarray(target, dtype=float))
        quad = 0.5 * a ** 2
        lin = self.delta * (a - 0.5 * self.delta)
        return float(np.mean(np.where(a <= self.delta, quad, lin)))


def default_losses() -> Registry:
    losses = Registry("loss")
    losses.register("mse")(MSE)
    losses.register("huber")(Huber)
    return losses


class Trainer:
    def __init__(self, step_fn, callbacks=()):
        self.step_fn = step_fn
        self.callbacks = list(callbacks)
        self.history: list[float] = []
        self.stop = False

    def fit(self, epochs: int):
        self.stop = False
        for epoch in range(epochs):
            loss = float(self.step_fn(epoch))
            self.history.append(loss)
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, loss)
            if self.stop:
                break
        return self.history


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best = float("inf")
        self.best_epoch = -1
        self.wait = 0

    def on_epoch_end(self, trainer, epoch, loss):
        if loss < self.best - self.min_delta:
            self.best, self.best_epoch, self.wait = loss, epoch, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.stop = True


class History:
    def __init__(self):
        self.records: list[tuple[int, float]] = []

    def on_epoch_end(self, trainer, epoch, loss):
        self.records.append((epoch, loss))
