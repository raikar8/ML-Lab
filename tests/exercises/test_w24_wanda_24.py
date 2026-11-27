import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_masks():
    torch.manual_seed(0)
    W = torch.randn(16, 64)
    X = torch.randn(256, 64); X[:, :4] *= 20  # outlier features
    mm, wm = m.magnitude_mask(W, 0.5), m.wanda_mask(W, X, 0.5)
    assert mm.sum(1).eq(32).all() and wm.sum(1).eq(32).all()
    assert m.pruned_error(W, wm, X) < m.pruned_error(W, mm, X)
    nm = m.nm_mask(W.abs() * X.norm(dim=0), 2, 4)
    assert nm.reshape(16, 16, 4).sum(-1).eq(2).all()
