import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def synthetic(seed=0, U=60, I=80, k=3):
    rng = np.random.default_rng(seed)
    P, Q = rng.normal(size=(U, k)), rng.normal(size=(I, k))
    R = np.clip(3 + P @ Q.T + 0.3 * rng.normal(size=(U, I)), 1, 5)
    mask = rng.random((U, I)) < 0.3
    u, i = np.nonzero(mask)
    t = np.c_[u, i, R[u, i]]
    rng.shuffle(t)
    return t[: int(0.8 * len(t))], t[int(0.8 * len(t)):], U, I


def test_mf_learns():
    tr, te, U, I = synthetic()
    mf = m.MF(U, I, k=8, lr=0.02, reg=0.05, epochs=40).fit(tr)
    base = np.sqrt(((te[:, 2] - tr[:, 2].mean()) ** 2).mean())
    assert mf.history[-1] < mf.history[0] and mf.rmse(te) < 0.85 * base
    r = m.recall_at_k(mf, tr, te, 10)
    assert 0 < r <= 1
