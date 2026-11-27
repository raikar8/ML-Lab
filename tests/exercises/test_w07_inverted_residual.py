import torch
import torch.nn as nn
from helpers import key_of, load

m = load(key_of(__file__))


def test_block_structure():
    b = m.InvertedResidual(16, 16, 1, 6)
    assert b.use_res and b(torch.randn(2, 16, 8, 8)).shape == (2, 16, 8, 8)
    b2 = m.InvertedResidual(16, 24, 2, 6)
    assert not b2.use_res and b2(torch.randn(2, 16, 8, 8)).shape == (2, 24, 4, 4)
    convs = [c for c in b.modules() if isinstance(c, nn.Conv2d)]
    assert [c.groups for c in convs] == [1, 96, 1] and all(c.bias is None for c in convs)
    assert len([c for c in m.InvertedResidual(8, 8, 1, 1).modules() if isinstance(c, nn.Conv2d)]) == 2


def test_counts():
    b = m.InvertedResidual(16, 16, 1, 6)
    assert m.count_params(b) == 16 * 96 + 96 * 9 + 96 * 16 + 2 * (96 + 96 + 16)
    assert m.count_macs(b, (1, 16, 8, 8)) == 64 * (16 * 96 + 96 * 9 + 96 * 16)
    lin = nn.Sequential(nn.Flatten(), nn.Linear(12, 5))
    assert m.count_macs(lin, (3, 12)) == 60
