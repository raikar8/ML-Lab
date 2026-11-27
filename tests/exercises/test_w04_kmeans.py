import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def blobs(seed=0):
    rng = np.random.default_rng(seed)
    c = np.array([[0, 0], [8, 0], [0, 8], [8, 8]], float)
    y = rng.integers(0, 4, 400)
    return c[y] + rng.normal(size=(400, 2)), c


def test_pp_spreads_centres():
    X, _ = blobs()
    C = m.kmeans_pp_init(X, 4, np.random.default_rng(3))
    d = ((C[:, None] - C[None]) ** 2).sum(-1) + np.eye(4) * 1e9
    assert C.shape == (4, 2) and d.min() > 9


def test_kmeans_recovers_centres():
    X, c = blobs(1)
    C, labels, inertia = m.kmeans(X, 4, np.random.default_rng(0))
    for cc in c:
        assert np.min(((C - cc) ** 2).sum(1)) < 0.2
    assert labels.shape == (400,) and inertia < 400 * 2.5
