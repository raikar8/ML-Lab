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
    raise NotImplementedError  # TODO: implement


class LogisticRegression:
    def __init__(self, lr=0.1, l2=0.0, epochs=500):
        raise NotImplementedError  # TODO: implement

    def predict_proba(self, X):
        raise NotImplementedError  # TODO: implement

    def predict(self, X, threshold=0.5):
        raise NotImplementedError  # TODO: implement

    def loss(self, X, y):
        raise NotImplementedError  # TODO: implement

    def fit(self, X, y):
        raise NotImplementedError  # TODO: implement
