import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def test_gini():
    assert m.gini(np.array([5, 5])) == 0.5 and m.gini(np.array([3, 0])) == 0.0


def test_best_split_bruteforce():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 20, 60).astype(float)
    y = (x + rng.normal(0, 3, 60) > 10).astype(int)
    t, sc = m.best_split(x, y, 2)
    best = np.inf
    for v in np.unique(x)[:-1]:
        l, r = y[x <= v], y[x > v]
        g = lambda a: 1 - ((np.bincount(a, minlength=2) / len(a)) ** 2).sum()
        best = min(best, (len(l) * g(l) + len(r) * g(r)) / len(y))
    assert abs(sc - best) < 1e-12 and (x <= t).sum() > 0
    assert m.best_split(np.ones(5), np.array([0, 1, 0, 1, 0]), 2)[0] is None


def test_tree_boxes():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (400, 2))
    y = ((X[:, 0] > 0.2) | (X[:, 1] < -0.4)).astype(int)
    tree = m.DecisionTree(max_depth=3).fit(X, y)
    assert (tree.predict(X) == y).mean() > 0.95
    stump = m.DecisionTree(max_depth=0).fit(X, y)
    assert len(set(stump.predict(X))) == 1
