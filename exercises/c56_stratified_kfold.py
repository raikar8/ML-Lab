"""
Stratified k-fold cross-validation from scratch

Write `stratified_kfold(y, k, seed=0)` returning a list of `k` pairs `(train_idx, val_idx)` (sorted NumPy int
arrays) such that

* every index appears in exactly one validation fold, and train = all other indices;
* for every class, its count in each validation fold differs by at most 1 across folds;
* the total size of the validation folds differs by at most 1 across folds;
* results are reproducible for a given `seed` (use `np.random.default_rng(seed)`).

Also write `cross_val_score(model_fn, X, y, k, seed=0)` that, for each fold, calls
`model_fn(X_train, y_train)` → a `predict(X)` callable, and returns the list of validation accuracies.
Raise `ValueError` if `k < 2` or `k > len(y)`.
"""

import numpy as np

def stratified_kfold(y, k: int, seed: int = 0):
    raise NotImplementedError  # TODO: implement


def cross_val_score(model_fn, X, y, k: int, seed: int = 0) -> list[float]:
    raise NotImplementedError  # TODO: implement
