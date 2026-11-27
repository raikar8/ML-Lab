import torch
from helpers import key_of, load

m = load(key_of(__file__))
V = m.Value


def test_against_torch():
    a, b = V(-4.0), V(2.0)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f + (a / 7).tanh() + (b.exp() / 10).log() + 1 - a
    g.backward()

    at, bt = torch.tensor(-4.0, dtype=torch.double, requires_grad=True), torch.tensor(2.0, dtype=torch.double, requires_grad=True)
    c = at + bt
    d = at * bt + bt ** 3
    c = c + c + 1
    c = c + 1 + c + (-at)
    d = d + d * 2 + (bt + at).relu()
    d = d + 3 * d + (bt - at).relu()
    e = c - d
    f = e ** 2
    gt = f / 2.0
    gt = gt + 10.0 / f + torch.tanh(at / 7) + torch.log(torch.exp(bt) / 10) + 1 - at
    gt.backward()
    assert abs(g.data - gt.item()) < 1e-9
    assert abs(a.grad - at.grad.item()) < 1e-6 and abs(b.grad - bt.grad.item()) < 1e-6


def test_reuse_accumulates():
    x = V(3.0)
    y = x * x + x
    y.backward()
    assert x.grad == 7.0
