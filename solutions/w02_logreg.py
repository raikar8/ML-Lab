"""
title: Logistic regression with L2, trained by gradient descent (NumPy)
week: 2
day: 3
level: easy
time: 30
tags: numpy, linear-models, optimisation
core: yes

Implement `LogisticRegression(lr=0.1, l2=0.0, epochs=500)` with:

* `fit(X, y)` — full-batch gradient descent on mean binary cross-entropy + `0.5 * l2 * ||w||²`
  (don't regularise the bias). Store `w` (shape `(d,)`) and `b` (float). Use a numerically stable sigmoid.
* `predict_proba(X)` → probabilities `(n,)`; `predict(X, threshold=0.5)` → 0/1 ints.
* `loss(X, y)` → the regularised objective.

Tip: the gradient is `X.T @ (p - y) / n + l2 * w`.
"""
import numpy as np


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegression:
    def __init__(self, lr=0.1, l2=0.0, epochs=500):
        self.lr, self.l2, self.epochs = lr, l2, epochs
        self.w, self.b = None, 0.0

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def loss(self, X, y):
        p = np.clip(self.predict_proba(X), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * self.l2 * self.w @ self.w)

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            p = self.predict_proba(X)
            gw = X.T @ (p - y) / n + self.l2 * self.w
            gb = float(np.mean(p - y))
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self
