import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def test_pca():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, 2)) * [5, 1]
    R = np.linalg.qr(rng.normal(size=(4, 4)))[0][:2]
    X = Z @ R + 3 + 0.01 * rng.normal(size=(500, 4))
    p = m.PCA(2).fit(X)
    assert p.components_.shape == (2, 4) and p.explained_variance_ratio_.sum() > 0.999
    assert abs(abs(p.components_[0] @ R[0]) - 1) < 1e-3
    np.testing.assert_allclose(p.inverse_transform(p.transform(X)), X, atol=0.05)


def test_low_rank_eckart_young():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 20))
    S = np.linalg.svd(A, compute_uv=False)
    err = np.linalg.norm(A - m.low_rank(A, 5))
    assert abs(err - np.sqrt((S[5:] ** 2).sum())) < 1e-9
    k = m.choose_rank(A, 0.9)
    assert (S[:k] ** 2).sum() / (S ** 2).sum() >= 0.9 > (S[:k - 1] ** 2).sum() / (S ** 2).sum()
