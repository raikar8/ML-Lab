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
        self.k, self.lr, self.reg, self.epochs = k, lr, reg, epochs
        rng = np.random.default_rng(seed)
        self.P = rng.normal(0, 0.1, (n_users, k))
        self.Q = rng.normal(0, 0.1, (n_items, k))
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        self.rng = rng
        self.mu = 0.0

    def fit(self, triples):
        t = np.asarray(triples, dtype=float)
        self.mu = float(t[:, 2].mean())
        self.history = []
        for _ in range(self.epochs):
            for idx in self.rng.permutation(len(t)):
                u, i, r = int(t[idx, 0]), int(t[idx, 1]), t[idx, 2]
                e = r - (self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])
                self.bu[u] += self.lr * (e - self.reg * self.bu[u])
                self.bi[i] += self.lr * (e - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (e * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (e * pu - self.reg * self.Q[i])
            self.history.append(self.rmse(t))
        return self

    def predict(self, u, i):
        u, i = np.asarray(u, dtype=int), np.asarray(i, dtype=int)
        return self.mu + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(-1)

    def rmse(self, triples):
        t = np.asarray(triples, dtype=float)
        return float(np.sqrt(((self.predict(t[:, 0], t[:, 1]) - t[:, 2]) ** 2).mean()))


def recall_at_k(model, train, test, k):
    train, test = np.asarray(train), np.asarray(test)
    n_items = model.Q.shape[0]
    recalls = []
    for u in np.unique(test[:, 0]).astype(int):
        rel = set(test[(test[:, 0] == u) & (test[:, 2] >= 4), 1].astype(int))
        if not rel:
            continue
        seen = set(train[train[:, 0] == u, 1].astype(int))
        cand = np.array([i for i in range(n_items) if i not in seen])
        top = cand[np.argsort(-model.predict(np.full(len(cand), u), cand))[:k]]
        recalls.append(len(rel & set(top.tolist())) / len(rel))
    return float(np.mean(recalls)) if recalls else 0.0
