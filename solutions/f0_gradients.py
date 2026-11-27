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
    x = np.array(x, dtype=float)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = f(x)
        x[i] = old - h
        fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g


def rel_error(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-12))


def logistic_loss_grad(w, X, y):
    z = X @ w
    # BCE = log(1 + e^z) − y z, computed stably
    loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    p = np.exp(-np.logaddexp(0.0, -z))              # σ(z) without overflow
    grad = X.T @ (p - y) / len(y)
    return loss, grad


def gradient_descent(f_grad, x0, lr: float, steps: int):
    x = np.array(x0, dtype=float)
    v, g = f_grad(x)
    hist = [float(v)]
    for _ in range(steps):
        x = x - lr * g
        v, g = f_grad(x)
        hist.append(float(v))
    return x, hist


def max_stable_lr(H) -> float:
    return 2.0 / float(np.linalg.eigvalsh(np.asarray(H, dtype=float)).max())
