import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def test_log_gauss_matches_formula():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3)); S = A @ A.T + np.eye(3)
    mu = rng.normal(size=3); X = rng.normal(size=(5, 3))
    D = X - mu
    ref = -0.5 * np.einsum("ni,ij,nj->n", D, np.linalg.inv(S), D) - 0.5 * np.log(np.linalg.det(2 * np.pi * S))
    np.testing.assert_allclose(m.log_gauss(X, mu, S), ref, atol=1e-10)


def test_em_monotone_and_recovers():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 300),
                   rng.multivariate_normal([5, 5], [[1, -0.5], [-0.5, 1]], 200)])
    pi, mus, S, ll = m.fit_gmm(X, 2, np.random.default_rng(2), n_iter=60)
    assert all(b >= a - 1e-8 for a, b in zip(ll, ll[1:]))
    order = np.argsort(mus[:, 0])
    np.testing.assert_allclose(mus[order], [[0, 0], [5, 5]], atol=0.25)
    np.testing.assert_allclose(np.sort(pi), [0.4, 0.6], atol=0.05)
    resp, _ = m.e_step(X, pi, mus, S)
    np.testing.assert_allclose(resp.sum(1), 1.0)
