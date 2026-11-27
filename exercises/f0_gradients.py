"""
title: Gradients you can trust — numerical gradients, gradient checking, gradient descent
week: 0
day: 3
level: easy
time: 35
tags: calculus, optimisation, numpy, foundations
core: yes

* `numerical_grad(f, x, h=1e-5)` — central-difference gradient of a scalar function `f` at array `x` (any shape);
  don't modify `x`.
* `rel_error(a, b)` — `‖a − b‖ / max(‖a‖, ‖b‖, 1e-12)`.
* `logistic_loss_grad(w, X, y)` — mean binary cross-entropy of `σ(Xw)` and its analytic gradient
  `(Xᵀ(σ(Xw) − y)) / n`; numerically stable for large |Xw| (use `np.logaddexp`).
* `gradient_descent(f_grad, x0, lr, steps)` — `f_grad(x) -> (value, grad)`; return `(x_final, history_of_values)`
  where the history has `steps + 1` entries (including the start).
* `max_stable_lr(H)` — for a quadratic with symmetric positive-definite Hessian `H`, the supremum learning rate for
  which gradient descent converges: `2 / λ_max(H)` (`np.linalg.eigvalsh` allowed here).
"""

import numpy as np

def numerical_grad(f, x, h: float = 1e-5):
    raise NotImplementedError  # TODO: implement


def rel_error(a, b) -> float:
    raise NotImplementedError  # TODO: implement


def logistic_loss_grad(w, X, y):
    raise NotImplementedError  # TODO: implement


def gradient_descent(f_grad, x0, lr: float, steps: int):
    raise NotImplementedError  # TODO: implement


def max_stable_lr(H) -> float:
    raise NotImplementedError  # TODO: implement
