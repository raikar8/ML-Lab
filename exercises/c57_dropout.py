"""
title: Inverted dropout — forward, backward, train vs eval
pattern: ML from scratch
level: easy
time: 20
frequency: high
week: 5
day: 5
tags: numpy, regularisation, backprop
complexity: O(n) time and memory (the mask is kept for backward)
hint: Train: mask ~ Bernoulli(1 − p), output x · mask / (1 − p) so the expected value is unchanged.
hint: Backward multiplies the upstream gradient by the same mask / (1 − p). Eval mode is the identity.
follow-up: Why "inverted" dropout instead of scaling by (1 − p) at test time? (no change needed in the deployed model)
follow-up: Why does dropout interact badly with BatchNorm, and why do most LLMs now use little or no dropout?

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
