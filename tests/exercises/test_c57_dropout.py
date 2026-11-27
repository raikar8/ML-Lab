import numpy as np
import pytest

from helpers import key_of, load

m = load(key_of(__file__))


def test_expectation_and_rate():
    d = m.Dropout(0.3, seed=0)
    x = np.ones((400, 500))
    y = d.forward(x)
    kept = y != 0
    assert abs(kept.mean() - 0.7) < 0.01
    assert np.allclose(y[kept], 1 / 0.7)
    assert abs(y.mean() - 1.0) < 0.01


def test_eval_and_edges():
    x = np.random.default_rng(1).normal(size=(5, 6))
    assert np.array_equal(m.Dropout(0.5).forward(x, train=False), x)
    assert np.array_equal(m.Dropout(0.0).forward(x), x)
    d = m.Dropout(1.0)
    assert np.all(d.forward(x) == 0) and np.all(d.backward(np.ones_like(x)) == 0)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            m.Dropout(bad)


def test_backward_matches_numerical():
    d = m.Dropout(0.4, seed=3)
    x = np.random.default_rng(2).normal(size=(4, 7))
    y = d.forward(x)
    g = np.random.default_rng(4).normal(size=y.shape)
    gx = d.backward(g)
    mask = y / np.where(x == 0, 1, x)  # the effective linear map per element
    assert np.allclose(gx, g * mask)


def test_generator_advances():
    d = m.Dropout(0.5, seed=7)
    a, b = d.forward(np.ones(1000)), d.forward(np.ones(1000))
    assert not np.array_equal(a, b)
    d2 = m.Dropout(0.5, seed=7)
    assert np.array_equal(d2.forward(np.ones(1000)), a)
