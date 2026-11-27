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
    d = X.shape[1]
    L = np.linalg.cholesky(Sigma)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (z * z).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)


def e_step(X, pi, mus, Sigmas):
    lp = np.stack([np.log(pi[k]) + log_gauss(X, mus[k], Sigmas[k]) for k in range(len(pi))], 1)
    m = lp.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(lp - m).sum(1))
    return np.exp(lp - lse[:, None]), float(lse.sum())


def m_step(X, resp, reg=1e-6):
    n, d = X.shape
    Nk = resp.sum(0)
    pi = Nk / n
    mus = (resp.T @ X) / Nk[:, None]
    Sigmas = np.empty((len(Nk), d, d))
    for k in range(len(Nk)):
        D = X - mus[k]
        Sigmas[k] = (resp[:, k, None] * D).T @ D / Nk[k] + reg * np.eye(d)
    return pi, mus, Sigmas


def fit_gmm(X, K, rng, n_iter=100):
    n, d = X.shape
    mus = X[rng.choice(n, K, replace=False)]
    Sigmas = np.stack([np.cov(X.T) + 1e-6 * np.eye(d)] * K)
    pi = np.full(K, 1 / K)
    logliks = []
    for _ in range(n_iter):
        resp, ll = e_step(X, pi, mus, Sigmas)
        logliks.append(ll)
        pi, mus, Sigmas = m_step(X, resp)
    return pi, mus, Sigmas, logliks
