import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_route_weights_use_unbiased_scores():
    logits = torch.tensor([[2.0, 0.0, -1.0, 1.0]])
    bias = torch.tensor([-10.0, 0.0, 10.0, 0.0])
    idx, w = m.route(logits, bias, 2)
    assert set(idx[0].tolist()) == {2, 3}
    s = torch.sigmoid(logits)[0]
    torch.testing.assert_close(w[0], torch.stack([s[i] for i in idx[0]]) / (s[2] + s[3]))


def test_bias_update_and_balancing():
    idx = torch.tensor([[0, 1], [0, 1], [0, 2]])
    torch.testing.assert_close(m.update_bias(torch.zeros(4), idx, 4, 0.1), torch.tensor([-0.1, -0.1, 0.1, 0.1]))
    g = torch.Generator().manual_seed(0)
    skew = torch.linspace(2, -2, 16)
    ratios = m.simulate(lambda t: torch.randn(512, 16, generator=g) + skew, 16, 2, 300, 0.01)
    assert ratios[0] > 2.5 and sum(ratios[-20:]) / 20 < 1.4
