import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_fake_quant_forward_backward():
    x = torch.tensor([-2.0, -0.26, 0.04, 0.31, 1.9], requires_grad=True)
    y = m.fake_quant(x, 0.1, qmin=-8, qmax=7)
    torch.testing.assert_close(y, torch.tensor([-0.8, -0.3, 0.0, 0.3, 0.7]))
    y.sum().backward()
    assert x.grad.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]


def test_round_ste():
    x = torch.tensor([0.4, 1.6], requires_grad=True)
    (m.round_ste(x) * torch.tensor([2.0, 3.0])).sum().backward()
    assert x.grad.tolist() == [2.0, 3.0]
    assert torch.round(x).sum().requires_grad  # plain round has zero grad; check STE differs:
    x.grad = None
    torch.round(x).sum().backward()
    assert x.grad.abs().sum() == 0
