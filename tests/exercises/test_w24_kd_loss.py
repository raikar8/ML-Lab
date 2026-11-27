import torch
import torch.nn.functional as F
from helpers import key_of, load

m = load(key_of(__file__))


def test_kd():
    torch.manual_seed(0)
    s = torch.randn(8, 5, requires_grad=True); t = torch.randn(8, 5, requires_grad=True)
    y = torch.randint(0, 5, (8,))
    assert abs(m.kd_loss(s, s.detach(), y, alpha=0.0).item()) < 1e-6
    torch.testing.assert_close(m.kd_loss(s, t, y, alpha=1.0), F.cross_entropy(s, y))
    m.kd_loss(s, t, y).backward()
    assert t.grad is None and s.grad is not None
    g = []
    for T in (2.0, 8.0):
        ss = s.detach().clone().requires_grad_()
        m.kd_loss(ss, t, y, T=T, alpha=0.0).backward(); g.append(ss.grad.norm())
    assert 0.2 < float(g[1] / g[0]) < 5  # T² keeps gradient magnitude comparable
