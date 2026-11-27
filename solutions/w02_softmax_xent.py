"""
title: Stable softmax, log-softmax and cross-entropy with gradient (NumPy)
week: 2
day: 2
level: easy
time: 25
tags: numpy, numerics, derivation
core: yes

Implement for a batch of logits `z` with shape `(N, K)` and integer labels `y` with shape `(N,)`:

* `logsumexp(z)` → `(N,)`, stable for logits like 1000 or −1000 (subtract the row max).
* `softmax(z)` and `log_softmax(z)` → `(N, K)`, stable.
* `cross_entropy(z, y)` → `(loss, dz)`: the **mean** negative log-likelihood and its gradient w.r.t. `z`,
  which is `(softmax(z) − onehot(y)) / N`.

No loops over samples; NumPy only.
"""
import numpy as np


def logsumexp(z):
    m = z.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))[:, 0]


def log_softmax(z):
    return z - logsumexp(z)[:, None]


def softmax(z):
    return np.exp(log_softmax(z))


def cross_entropy(z, y):
    n = z.shape[0]
    lp = log_softmax(z)
    loss = -lp[np.arange(n), y].mean()
    dz = np.exp(lp)
    dz[np.arange(n), y] -= 1.0
    return loss, dz / n
