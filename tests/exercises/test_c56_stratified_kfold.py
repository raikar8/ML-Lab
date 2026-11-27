import numpy as np
import pytest

from helpers import key_of, load

m = load(key_of(__file__))


def check(y, k, seed=0):
    folds = m.stratified_kfold(y, k, seed)
    y = np.asarray(y)
    assert len(folds) == k
    allv = np.concatenate([v for _, v in folds])
    assert sorted(allv.tolist()) == list(range(len(y)))
    for tr, va in folds:
        assert np.all(np.diff(va) > 0) and np.all(np.diff(tr) > 0)
        assert set(tr.tolist()) | set(va.tolist()) == set(range(len(y))) and not set(tr) & set(va)
    sizes = [len(v) for _, v in folds]
    assert max(sizes) - min(sizes) <= 1
    for c in np.unique(y):
        cnt = [int(np.sum(y[v] == c)) for _, v in folds]
        assert max(cnt) - min(cnt) <= 1, (c, cnt)
    return folds


def test_properties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(5, 60))
        y = rng.integers(0, int(rng.integers(1, 5)), size=n)
        check(y, int(rng.integers(2, min(n, 8) + 1)))
    check(["a"] * 7 + ["b"] * 3, 3)


def test_reproducible_and_seed_dependent():
    y = np.repeat([0, 1, 2], 20)
    a = m.stratified_kfold(y, 5, seed=1)
    b = m.stratified_kfold(y, 5, seed=1)
    c = m.stratified_kfold(y, 5, seed=2)
    assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b))
    assert not all(np.array_equal(x[1], z[1]) for x, z in zip(a, c))


def test_errors_and_cross_val():
    with pytest.raises(ValueError):
        m.stratified_kfold([0, 1, 0], 1)
    with pytest.raises(ValueError):
        m.stratified_kfold([0, 1, 0], 4)
    X = np.concatenate([np.arange(20), 100 + np.arange(20)]).reshape(-1, 1)   # two well-separated classes
    y = (X[:, 0] >= 100).astype(int)

    def fit(Xt, yt):
        thr = (Xt[yt == 0].max() + Xt[yt == 1].min()) / 2
        return lambda Xv: (Xv[:, 0] > thr).astype(int)

    s = m.cross_val_score(fit, X, y, 4)
    assert len(s) == 4 and all(v == 1.0 for v in s)
