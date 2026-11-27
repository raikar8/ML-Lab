import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_streaming_and_export():
    torch.manual_seed(0)
    net = m.StreamingMaskNet(40, 32).eval()
    x = torch.randn(1, 50, 40)
    with torch.no_grad():
        full, _ = net(x, net.init_state(1))
        h, outs = net.init_state(1), []
        for i in range(0, 50, 10):
            o, h = net(x[:, i:i + 10], h); outs.append(o)
    torch.testing.assert_close(torch.cat(outs, 1), full)
    ep = m.export_step(net, 10)
    o, h2 = ep.module()(x[:, :10], net.init_state(1))
    torch.testing.assert_close(o, full[:, :10], atol=1e-5, rtol=1e-5)
    assert not [n for n, _ in net.named_buffers()]
