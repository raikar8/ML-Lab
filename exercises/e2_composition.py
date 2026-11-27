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

@runtime_checkable
class Stage(Protocol):
    def process(self, frame: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...

class Gain:
    def __init__(self, db: float):
        raise NotImplementedError  # TODO: implement

    def process(self, frame):
        raise NotImplementedError  # TODO: implement

    def reset(self):
        raise NotImplementedError  # TODO: implement


class DCBlock:
    def __init__(self, r: float = 0.995):
        raise NotImplementedError  # TODO: implement

    def reset(self):
        raise NotImplementedError  # TODO: implement

    def process(self, frame):
        raise NotImplementedError  # TODO: implement


class Clip:
    def __init__(self, limit: float):
        raise NotImplementedError  # TODO: implement

    def process(self, frame):
        raise NotImplementedError  # TODO: implement

    def reset(self):
        raise NotImplementedError  # TODO: implement


class Pipeline:
    def __init__(self, stages):
        raise NotImplementedError  # TODO: implement

    def process(self, frame):
        raise NotImplementedError  # TODO: implement

    def reset(self):
        raise NotImplementedError  # TODO: implement

    def then(self, stage) -> "Pipeline":
        raise NotImplementedError  # TODO: implement
