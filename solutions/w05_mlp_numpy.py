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
    return {"W1": rng.normal(0, np.sqrt(2 / D), (D, H)), "b1": np.zeros(H),
            "W2": rng.normal(0, np.sqrt(1 / H), (H, K)), "b2": np.zeros(K)}


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.maximum(z1, 0)
    logits = a1 @ params["W2"] + params["b2"]
    return logits, (X, z1, a1)


def loss_and_grads(params, X, y):
    logits, (X, z1, a1) = forward(params, X)
    n = len(y)
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    loss = -logp[np.arange(n), y].mean()
    dlogits = np.exp(logp)
    dlogits[np.arange(n), y] -= 1
    dlogits /= n
    grads = {"W2": a1.T @ dlogits, "b2": dlogits.sum(0)}
    da1 = dlogits @ params["W2"].T
    dz1 = da1 * (z1 > 0)
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(0)
    return float(loss), grads


def train(X, y, H, K, lr, steps, rng):
    params = init_params(X.shape[1], H, K, rng)
    for _ in range(steps):
        _, g = loss_and_grads(params, X, y)
        for k in params:
            params[k] -= lr * g[k]
    return params
