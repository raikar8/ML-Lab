import numpy as np
import pytest

from helpers import key_of, load

m = load(key_of(__file__))


def test_bayes_base_rate():
    post = m.bayes_update({"sick": 0.001, "healthy": 0.999}, {"sick": 0.99, "healthy": 0.01})
    assert abs(post["sick"] - 0.0902) < 1e-3 and abs(sum(post.values()) - 1) < 1e-12
    with pytest.raises(ValueError):
        m.bayes_update({"a": 1.0}, {"a": 0.0})


def test_sample_categorical():
    rng = np.random.default_rng(0)
    p = [0.1, 0.0, 0.6, 0.3]
    s = m.sample_categorical(p, 200_000, rng)
    assert s.dtype.kind == "i" and set(np.unique(s)) <= {0, 2, 3}
    assert np.allclose(np.bincount(s, minlength=4) / len(s), p, atol=0.005)


def test_gaussian_mle():
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    mu, v, vu = m.gaussian_mle(x)
    assert mu == 5.0 and v == 4.0 and np.isclose(vu, 32 / 7)
    assert np.isnan(m.gaussian_mle([3.0])[2])


def test_bootstrap_covers_mean_and_is_reproducible():
    rng = np.random.default_rng(1)
    hits = 0
    for t in range(60):
        x = rng.normal(10, 2, size=60)
        lo, hi = m.bootstrap_ci(x, np.mean, n_boot=800, seed=t)
        hits += lo <= 10 <= hi
        assert hi - lo < 2.0
    assert hits >= 50                                   # ~95% nominal coverage
    x = rng.normal(size=30)
    assert m.bootstrap_ci(x, np.median, seed=3) == m.bootstrap_ci(x, np.median, seed=3)


def test_binomial_tail():
    assert abs(m.binomial_tail(10, 8) - 56 / 1024) < 1e-12
    assert m.binomial_tail(5, 0) == pytest.approx(1.0) and m.binomial_tail(5, 6) == 0
