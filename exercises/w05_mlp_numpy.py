"""
title: Two-layer MLP with manual backprop (NumPy)
week: 5
day: 2
level: medium
time: 40
tags: numpy, backprop, classic
core: yes

A classifier `X (N,D) → Linear(D,H) → ReLU → Linear(H,K) → softmax cross-entropy (mean)`.

* `init_params(D, H, K, rng)` → dict with `W1 (D,H)`, `b1 (H,)`, `W2 (H,K)`, `b2 (K,)` (He init for W1, zeros for biases).
* `forward(params, X)` → `(logits, cache)`.
* `loss_and_grads(params, X, y)` → `(loss, grads)` with `grads` having the same keys/shapes as `params`, computed by
  hand (no autograd). Use a stable softmax.
* `train(X, y, H, K, lr, steps, rng)` → params after `steps` of full-batch gradient descent.
"""

import numpy as np

def init_params(D, H, K, rng):
    raise NotImplementedError  # TODO: implement


def forward(params, X):
    raise NotImplementedError  # TODO: implement


def loss_and_grads(params, X, y):
    raise NotImplementedError  # TODO: implement


def train(X, y, H, K, lr, steps, rng):
    raise NotImplementedError  # TODO: implement
