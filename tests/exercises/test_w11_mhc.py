import torch
import torch.nn as nn
from helpers import key_of, load

m = load(key_of(__file__))


def test_sinkhorn_doubly_stochastic_and_closed():
    torch.manual_seed(0)
    A, B = m.sinkhorn(torch.randn(4, 4) * 3, 50), m.sinkhorn(torch.randn(4, 4) * 3, 50)
    for P in (A, B, A @ B):
        torch.testing.assert_close(P.sum(0), torch.ones(4), atol=1e-4, rtol=0)
        torch.testing.assert_close(P.sum(1), torch.ones(4), atol=1e-4, rtol=0)
        assert (P >= 0).all() and torch.linalg.matrix_norm(P, 2) <= 1 + 1e-4
    assert m.sinkhorn(torch.randn(3, 4, 4)).shape == (3, 4, 4)


def test_stack_is_stable_and_trains():
    torch.manual_seed(1)
    d, n = 16, 4
    blocks = nn.ModuleList([m.HyperConnection(n, d, nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))) for _ in range(30)])
    x = m.expand(torch.randn(2, 5, d), n)
    assert x.shape == (2, 5, 4, 16)
    for b in blocks:
        x = b(x)
    out = m.collapse(x)
    assert out.shape == (2, 5, 16) and torch.isfinite(out).all() and out.abs().max() < 1e3
    out.sum().backward()
    assert blocks[0].res_logits.grad is not None and torch.isfinite(blocks[0].res_logits.grad).all()
