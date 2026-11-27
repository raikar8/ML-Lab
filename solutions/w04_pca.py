"""
title: PCA via SVD, explained variance, and low-rank reconstruction
week: 4
day: 3
level: easy
time: 25
tags: numpy, pca, linear-algebra, compression
core: yes

Implement `PCA(n_components)` with `fit(X)`, `transform(X)`, `inverse_transform(Z)` using the SVD of the centred
data (not the covariance matrix). Store `mean_`, `components_` (`(k, d)`, rows = principal directions) and
`explained_variance_ratio_` (`(k,)`). Also implement `low_rank(A, k)` → the best rank-k approximation (Eckart–Young)
and `choose_rank(A, energy)` → smallest k whose singular values retain at least `energy` of the total squared
Frobenius norm.
"""
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.k = n_components

    def fit(self, X):
        self.mean_ = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[: self.k]
        var = S ** 2
        self.explained_variance_ratio_ = var[: self.k] / var.sum()
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


def low_rank(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]


def choose_rank(A, energy):
    S = np.linalg.svd(A, compute_uv=False)
    frac = np.cumsum(S ** 2) / (S ** 2).sum()
    return int(np.searchsorted(frac, energy - 1e-12) + 1)
