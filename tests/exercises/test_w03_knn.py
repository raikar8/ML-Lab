import inspect

import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def test_dists():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(7, 5)), rng.normal(size=(9, 5))
    ref = ((A[:, None] - B[None]) ** 2).sum(-1)
    np.testing.assert_allclose(m.pairwise_sq_dists(A, B), ref, atol=1e-10)
    assert (m.pairwise_sq_dists(A, A) >= 0).all()
    assert "for " not in inspect.getsource(m.pairwise_sq_dists)


def test_knn_blobs():
    rng = np.random.default_rng(1)
    c = np.array([[0, 0], [5, 5], [0, 5]])
    y = rng.integers(0, 3, 300)
    X = c[y] + rng.normal(size=(300, 2))
    pred = m.knn_predict(X[:200], y[:200], X[200:], k=5)
    assert (pred == y[200:]).mean() > 0.95
