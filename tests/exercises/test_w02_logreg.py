import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 3))
    w = np.array([2.0, -1.0, 0.5])
    y = (rng.random(400) < 1 / (1 + np.exp(-(X @ w + 0.3)))).astype(float)
    return X, y


def test_sigmoid_stable():
    s = m.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0])


def test_fit_and_loss_decreases():
    X, y = data()
    clf = m.LogisticRegression(lr=0.5, epochs=1)
    clf.fit(X, y)
    l1 = clf.loss(X, y)
    clf = m.LogisticRegression(lr=0.5, epochs=400).fit(X, y)
    assert clf.loss(X, y) < l1
    assert (clf.predict(X) == y).mean() > 0.7
    assert np.sign(clf.w).tolist() == [1, -1, 1]


def test_l2_shrinks():
    X, y = data(1)
    a = m.LogisticRegression(lr=0.5, epochs=400).fit(X, y)
    b = m.LogisticRegression(lr=0.5, epochs=400, l2=1.0).fit(X, y)
    assert np.linalg.norm(b.w) < np.linalg.norm(a.w)
