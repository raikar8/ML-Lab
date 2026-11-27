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
    raise NotImplementedError  # TODO: implement


def sample_categorical(probs, n: int, rng) -> np.ndarray:
    raise NotImplementedError  # TODO: implement


def gaussian_mle(x):
    raise NotImplementedError  # TODO: implement


def bootstrap_ci(x, stat, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    raise NotImplementedError  # TODO: implement


def binomial_tail(n: int, k: int, p: float = 0.5) -> float:
    raise NotImplementedError  # TODO: implement
