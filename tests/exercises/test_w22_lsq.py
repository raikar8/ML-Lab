import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_quantizer_levels_and_grads():
    q = m.LSQQuantizer(bits=3, init_scale=0.5)
    x = torch.linspace(-3, 3, 50, requires_grad=True)
    y = q(x)
    assert torch.isin(torch.round(y.detach() / 0.5).unique(), torch.arange(-4, 4).float()).all()
    y.sum().backward()
    assert q.scale.grad is not None and q.scale.grad.abs() > 0 and x.grad[25] == 1


def test_qat_learns():
    torch.manual_seed(0)
    X = torch.randn(256, 8); W = torch.randn(3, 8); y = X @ W.T
    layer = m.QATLinear(8, 3, bits=4)
    opt = torch.optim.Adam(layer.parameters(), 0.02)
    for _ in range(300):
        opt.zero_grad(); loss = ((layer(X) - y) ** 2).mean(); loss.backward(); opt.step()
    assert loss.item() < 0.1 * (y ** 2).mean().item()
    assert layer.wq(layer.weight).detach().div(layer.wq.scale).round().unique().numel() <= 16
