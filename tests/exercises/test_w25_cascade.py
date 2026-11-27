import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def test_cascade():
    rng = np.random.default_rng(0)
    n_neg, n_pos = 200000, 2000
    neg_s1, neg_s2 = rng.beta(2, 8, n_neg), rng.beta(1.5, 12, n_neg)
    pos_s1, pos_s2 = rng.beta(8, 2, n_pos), rng.beta(10, 2, n_pos)
    assert m.cascade_decisions([0.9, 0.9, 0.1], [0.9, 0.1, 0.9], 0.5, 0.5).tolist() == [True, False, False]
    grid = np.linspace(0, 1, 21)
    t1, t2 = m.pick_thresholds(pos_s1, pos_s2, neg_s1, neg_s2, 24.0, 1.0, grid)
    acc = m.cascade_decisions(neg_s1, neg_s2, t1, t2)
    assert m.far_per_hour(acc, 24.0) <= 1.0
    assert m.frr(m.cascade_decisions(pos_s1, pos_s2, t1, t2)) < 0.2 and m.stage2_duty(neg_s1, t1) < 0.5
