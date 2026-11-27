import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def test_tree_fits_step():
    X = np.linspace(0, 1, 50)[:, None]
    y = (X[:, 0] > 0.5) * 3.0
    t = m.RegressionTree(max_depth=1).fit(X, y)
    np.testing.assert_allclose(t.predict(X), y)


def test_boosting_sine():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 6, (300, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=300)
    g = m.GradientBoostingRegressor(n_estimators=60, learning_rate=0.2, max_depth=2).fit(X, y)
    assert len(g.train_loss_) == 60
    assert all(b <= a + 1e-12 for a, b in zip(g.train_loss_, g.train_loss_[1:]))
    Xt = np.linspace(0.2, 5.8, 100)[:, None]
    assert np.mean((g.predict(Xt) - np.sin(Xt[:, 0])) ** 2) < 0.03
