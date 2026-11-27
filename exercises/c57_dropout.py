"""
Inverted dropout — forward, backward, train vs eval

Implement `Dropout(p, seed=0)` in NumPy:

* `forward(x, train=True)` — in training, zero each element independently with probability `p` and scale the
  survivors by `1 / (1 − p)`; in eval (`train=False`) return `x` unchanged. Draw the mask with
  `np.random.default_rng(seed)` created once in `__init__` (so calls advance the same generator).
* `backward(grad)` — gradient w.r.t. the input of the **last training-mode forward**.
* `p == 0` is the identity; `p == 1` outputs zeros (and zero gradient). Raise `ValueError` for `p` outside [0, 1].
"""

import numpy as np

class Dropout:
    def __init__(self, p: float, seed: int = 0):
        raise NotImplementedError  # TODO: implement

    def forward(self, x, train: bool = True):
        raise NotImplementedError  # TODO: implement

    def backward(self, grad):
        raise NotImplementedError  # TODO: implement
