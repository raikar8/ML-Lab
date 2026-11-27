"""
title: Probability in code — Bayes updates, sampling, MLE and the bootstrap
week: 0
day: 2
level: easy
time: 35
tags: probability, statistics, numpy, foundations
core: yes

* `bayes_update(prior, likelihood)` — `prior[h]` and `likelihood[h] = P(data | h)` are dicts over the same hypotheses;
  return the normalised posterior dict. Raise `ValueError` if the evidence is 0.
* `sample_categorical(probs, n, rng)` — draw `n` indices from the categorical distribution `probs` (sums to 1) by
  **inverse-CDF**: one `rng.random(n)` call, `np.cumsum`, `np.searchsorted`. Return an int array.
* `gaussian_mle(x)` — `(mu, var_mle, var_unbiased)` for a 1-D sample (var_mle divides by n, unbiased by n − 1;
  unbiased is `nan` for n = 1).
* `bootstrap_ci(x, stat, n_boot=2000, alpha=0.05, seed=0)` — percentile bootstrap CI `(lo, hi)` for `stat(sample)`:
  resample `len(x)` points with replacement `n_boot` times using `np.random.default_rng(seed)`, return the
  `alpha/2` and `1 − alpha/2` quantiles (`np.quantile`) of the statistic.
* `binomial_tail(n, k, p=0.5)` — `P(X >= k)` for `X ~ Binomial(n, p)`, exactly (`math.comb`).
"""
import math

import numpy as np


def bayes_update(prior: dict, likelihood: dict) -> dict:
    joint = {h: prior[h] * likelihood[h] for h in prior}
    z = sum(joint.values())
    if z <= 0:
        raise ValueError("evidence is zero: the data is impossible under every hypothesis")
    return {h: v / z for h, v in joint.items()}


def sample_categorical(probs, n: int, rng) -> np.ndarray:
    cdf = np.cumsum(np.asarray(probs, dtype=float))
    cdf[-1] = 1.0                                   # guard against round-off
    return np.searchsorted(cdf, rng.random(n), side="right").astype(int)


def gaussian_mle(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.mean()
    ss = ((x - mu) ** 2).sum()
    return float(mu), float(ss / n), float(ss / (n - 1)) if n > 1 else float("nan")


def bootstrap_ci(x, stat, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def binomial_tail(n: int, k: int, p: float = 0.5) -> float:
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(max(k, 0), n + 1))
