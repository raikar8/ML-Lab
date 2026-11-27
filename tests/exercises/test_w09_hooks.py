import pytest
import torch
import torch.nn as nn
from helpers import key_of, load

m = load(key_of(__file__))


def model():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    net[0].weight.requires_grad_(False)
    return net


def test_count():
    net = model()
    assert m.count_parameters(net) == 4 * 8 + 8 + 8 * 3 + 3
    assert m.count_parameters(net, trainable_only=True) == 8 + 8 * 3 + 3
    tied = nn.Sequential(net[2], net[2])
    assert m.count_parameters(tied) == 27


def test_recorder():
    net = model()
    x = torch.randn(16, 4)
    with m.ActivationRecorder(net) as rec:
        net(x); net(x * 2)
    assert set(rec.stats) == {"0", "2"} and rec.stats["0"]["calls"] == 2
    assert rec.stats["0"]["max"] == pytest.approx(float(net[0](x * 2).max().clamp_min(net[0](x).max())), rel=1e-5)
    assert all(len(mod._forward_hooks) == 0 for mod in net.modules())
    with pytest.raises(RuntimeError):
        with m.ActivationRecorder(net):
            raise RuntimeError
    assert all(len(mod._forward_hooks) == 0 for mod in net.modules())
