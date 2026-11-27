import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))
rng = np.random.default_rng(0)
y = (rng.random(300) < 0.2).astype(int)
s = y * 0.8 + rng.normal(size=300)


def test_prf():
    assert m.precision_recall_f1([1, 0, 1, 1], [1, 1, 0, 1]) == (2 / 3, 2 / 3, 2 / 3)
    assert m.precision_recall_f1([0, 0], [0, 0]) == (0.0, 0.0, 0.0)


def _auc_bruteforce(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])


def test_auc():
    assert abs(m.roc_auc(y, s) - _auc_bruteforce(y, s)) < 1e-12
    sr = np.round(s, 1)  # ties
    assert abs(m.roc_auc(y, sr) - _auc_bruteforce(y, sr)) < 1e-12


def test_ap():
    assert abs(m.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) - (1 + 2 / 3) / 2) < 1e-12
    assert m.average_precision(y, s) > y.mean()


def test_ece():
    assert m.ece([1, 0], [1.0, 0.0]) == 0.0
    assert abs(m.ece([1, 1, 0, 0], [0.9, 0.9, 0.9, 0.9]) - 0.4) < 1e-12
