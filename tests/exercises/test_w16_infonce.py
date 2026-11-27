import math

import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_clip_loss():
    torch.manual_seed(0)
    a = torch.randn(16, 8)
    assert m.clip_loss(a, a.clone(), 0.01).item() < 1e-3
    rnd = m.clip_loss(torch.randn(256, 8), torch.randn(256, 8), 1e6).item()
    assert abs(rnd - math.log(256)) < 1e-3
    a.requires_grad_()
    m.clip_loss(a, torch.randn(16, 8)).backward()
    assert a.grad is not None
