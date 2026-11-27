import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_stream_equals_offline():
    torch.manual_seed(0)
    for k, d in [(3, 1), (3, 4), (5, 2), (1, 1)]:
        layer = m.StreamingCausalConv1d(4, 6, k, dilation=d).double()
        x = torch.randn(2, 4, 57, dtype=torch.double)
        ref = layer(x)
        assert ref.shape == (2, 6, 57)
        layer.reset(2)
        outs, i = [], 0
        for t in [1, 7, 3, 16, 30]:
            outs.append(layer.step(x[..., i:i + t])); i += t
        torch.testing.assert_close(torch.cat(outs, -1), ref)


def test_causal():
    layer = m.StreamingCausalConv1d(1, 1, 3, 2)
    x = torch.randn(1, 1, 20)
    y1 = layer(x)
    x2 = x.clone(); x2[..., 10:] += 5
    torch.testing.assert_close(layer(x2)[..., :10], y1[..., :10])
