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
        raise NotImplementedError  # TODO: implement

    def fit(self, X):
        raise NotImplementedError  # TODO: implement

    def transform(self, X):
        raise NotImplementedError  # TODO: implement

    def inverse_transform(self, Z):
        raise NotImplementedError  # TODO: implement


def low_rank(A, k):
    raise NotImplementedError  # TODO: implement


def choose_rank(A, energy):
    raise NotImplementedError  # TODO: implement
