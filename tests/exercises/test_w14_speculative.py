import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_output_distribution_equals_target():
    torch.manual_seed(0)
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    q = torch.tensor([0.1, 0.2, 0.3, 0.4])
    counts = torch.zeros(4)
    N = 40000
    xs = torch.multinomial(q, N, replacement=True)
    us = torch.rand(N)
    for x, u in zip(xs.tolist(), us.tolist()):
        _, tok = m.accept_or_resample(p, q, x, u)
        counts[tok] += 1
    assert (counts / N - p).abs().max() < 0.012


def test_step_and_expected():
    p = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    q = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    assert m.speculative_step(p, q, torch.tensor([0, 1])) == [0, 1, 2]
    q2 = torch.tensor([[1.0, 0, 0], [0, 0, 1.0]])
    assert m.speculative_step(p, q2, torch.tensor([0, 2])) == [0, 1]
    assert abs(m.expected_tokens(0.8, 4) - 3.3616) < 1e-4 and m.expected_tokens(1, 3) == 4
