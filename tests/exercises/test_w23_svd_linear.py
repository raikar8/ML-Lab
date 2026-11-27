import torch
import torch.nn as nn
from helpers import key_of, load

m = load(key_of(__file__))


def test_svd():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(32, 8) @ torch.randn(8, 64) + 0.01 * torch.randn(32, 64))
    assert m.rank_for_energy(lin.weight.detach(), 0.99) <= 8
    comp = m.svd_compress(lin, 8)
    x = torch.randn(10, 64)
    assert (comp(x) - lin(x)).norm() / lin(x).norm() < 0.02
    assert m.saves_params(64, 32, 8) and not m.saves_params(64, 32, 30)
