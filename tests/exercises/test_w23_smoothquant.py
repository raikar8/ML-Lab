import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_outlier_fixes():
    torch.manual_seed(0)
    X = torch.randn(64, 128, dtype=torch.double)
    X[:, [3, 70]] *= 80
    W = torch.randn(32, 128, dtype=torch.double) * 0.05
    Y = X @ W.T
    err = lambda Yh: float((Y - Yh).norm() / Y.norm())
    base, sq, rot = err(m.w8a8(X, W)), err(m.smoothquant(X, W)), err(m.rotated(X, W))
    assert sq < 0.5 * base and rot < 0.5 * base
    H = m.hadamard(128)
    torch.testing.assert_close(H @ H.T, torch.eye(128, dtype=torch.double))
    s = m.smooth_scales(X, W)
    torch.testing.assert_close((X / s) @ (W * s).T, Y)
