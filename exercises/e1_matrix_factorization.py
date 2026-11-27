"""
title: Matrix factorisation with biases (SGD) and recall@k
week: 0
day: 2
level: medium
time: 40
tags: numpy, recsys, elective
core: no

* `MF(n_users, n_items, k=16, lr=0.02, reg=0.02, epochs=30, seed=0)` with `fit(triples)` where `triples` is an int/float
  array `(n, 3)` of `(user, item, rating)`: initialise factors `N(0, 0.1)`, biases 0, `mu` = mean rating; for each epoch shuffle
  and do per-sample SGD on `(r − (mu + b_u + b_i + p_u·q_i))²` with L2 `reg` on `p_u, q_i, b_u, b_i`. Store per-epoch train
  RMSE in `self.history`.
* `predict(u, i)` (arrays) and `rmse(triples)`.
* `recall_at_k(model, train, test, k)` — for each test user, rank all items they haven't rated in `train` by predicted score; recall =
  fraction of their test items (rating ≥ 4) in the top k, averaged over users that have any.
"""

import numpy as np

class MF:
    def __init__(self, n_users, n_items, k=16, lr=0.02, reg=0.02, epochs=30, seed=0):
        raise NotImplementedError  # TODO: implement

    def fit(self, triples):
        raise NotImplementedError  # TODO: implement

    def predict(self, u, i):
        raise NotImplementedError  # TODO: implement

    def rmse(self, triples):
        raise NotImplementedError  # TODO: implement


def recall_at_k(model, train, test, k):
    raise NotImplementedError  # TODO: implement
