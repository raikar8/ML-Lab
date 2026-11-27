"""
title: Composition over inheritance — a streaming audio pipeline of stateful stages
week: 0
day: 3
level: medium
time: 40
tags: oop, composition, protocol, audio, streaming, elective
core: yes

Design small classes that **compose**, all satisfying one `Stage` protocol (`process(frame) -> frame`, `reset()`), where a
frame is a 1-D NumPy float array:

* `Gain(db)` — multiply by `10 ** (db / 20)`; stateless.
* `DCBlock(r=0.995)` — first-order DC-blocking high-pass `y[n] = x[n] − x[n−1] + r·y[n−1]`; **stateful across frames**
  (remember the last input and output), so processing a signal frame by frame gives exactly the same output as
  processing it in one go. `reset()` clears the state.
* `Clip(limit)` — clamp to `[−limit, limit]`; also counts the total number of clipped samples in `clipped` (reset to 0
  by `reset()`).
* `Pipeline(stages)` — a **composite**: it is itself a `Stage`; `process` runs the stages in order, `reset` resets all
  of them; `Pipeline.then(stage)` returns a **new** pipeline with the stage appended (the original is unchanged).

`Stage` (given) is a `typing.Protocol` marked `@runtime_checkable`, so `isinstance(obj, Stage)` works for any of these
classes **without inheritance**. None of your classes may inherit from each other (or from `Stage`).
"""
from typing import Protocol, runtime_checkable

import numpy as np


# given
@runtime_checkable
class Stage(Protocol):
    def process(self, frame: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


class Gain:
    def __init__(self, db: float):
        self.g = 10.0 ** (db / 20.0)

    def process(self, frame):
        return np.asarray(frame, dtype=float) * self.g

    def reset(self):
        pass


class DCBlock:
    def __init__(self, r: float = 0.995):
        self.r = r
        self.reset()

    def reset(self):
        self.x1 = 0.0
        self.y1 = 0.0

    def process(self, frame):
        x = np.asarray(frame, dtype=float)
        y = np.empty_like(x)
        x1, y1, r = self.x1, self.y1, self.r
        for n, xn in enumerate(x):
            y1 = xn - x1 + r * y1
            x1 = xn
            y[n] = y1
        self.x1, self.y1 = x1, y1
        return y


class Clip:
    def __init__(self, limit: float):
        self.limit = limit
        self.clipped = 0

    def process(self, frame):
        x = np.asarray(frame, dtype=float)
        self.clipped += int(np.count_nonzero(np.abs(x) > self.limit))
        return np.clip(x, -self.limit, self.limit)

    def reset(self):
        self.clipped = 0


class Pipeline:
    def __init__(self, stages):
        self.stages = list(stages)

    def process(self, frame):
        for s in self.stages:
            frame = s.process(frame)
        return frame

    def reset(self):
        for s in self.stages:
            s.reset()

    def then(self, stage) -> "Pipeline":
        return Pipeline(self.stages + [stage])
