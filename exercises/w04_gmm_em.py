"""
title: EM for a Gaussian mixture (full covariance, log-space)
week: 4
day: 2
level: hard
time: 50
tags: numpy, em, probability
core: no

Implement EM for a K-component GMM with full covariances.

* `log_gauss(X, mu, Sigma)` → `(n,)` log-density of each row under N(mu, Sigma), via a Cholesky factor
  (no explicit inverse or determinant).
* `e_step(X, pi, mus, Sigmas)` → `(resp (n,K), loglik float)` computed with log-sum-exp.
* `m_step(X, resp, reg=1e-6)` → `(pi, mus, Sigmas)`; add `reg * I` to each covariance.
* `fit_gmm(X, K, rng, n_iter=100)` → `(pi, mus, Sigmas, logliks)`; initialise means from K random points, shared data
  covariance, uniform weights. `logliks` must be non-decreasing (up to 1e-8).
"""

import numpy as np

def log_gauss(X, mu, Sigma):
    raise NotImplementedError  # TODO: implement


def e_step(X, pi, mus, Sigmas):
    raise NotImplementedError  # TODO: implement


def m_step(X, resp, reg=1e-6):
    raise NotImplementedError  # TODO: implement


def fit_gmm(X, K, rng, n_iter=100):
    raise NotImplementedError  # TODO: implement
