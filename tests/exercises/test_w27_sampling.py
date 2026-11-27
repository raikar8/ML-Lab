import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_filters():
    lg = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    assert torch.isinf(m.top_k_filter(lg, 2))[0].tolist() == [False, False, True, True]
    assert torch.isinf(m.top_p_filter(lg, 0.7))[0].tolist() == [False, False, True, True]
    assert torch.isinf(m.top_p_filter(lg, 0.1))[0].tolist() == [False, True, True, True]
    shuffled = lg[:, [2, 0, 3, 1]]
    assert torch.isinf(m.top_p_filter(shuffled, 0.7))[0].tolist() == [True, False, True, False]
    rp = m.repetition_penalty(torch.tensor([[2.0, -2.0, 1.0]]), torch.tensor([[0, 1]]), 2.0)
    assert rp.tolist() == [[1.0, -4.0, 1.0]]


def test_sample():
    g = torch.Generator().manual_seed(0)
    lg = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]])).repeat(20000, 1)
    s = m.sample(lg, T=1.0, k=2, generator=g)
    frac = (s == 0).float().mean().item()
    assert set(s.unique().tolist()) == {0, 1} and abs(frac - 0.625) < 0.02
    assert (m.sample(lg[:3], T=0) == 0).all()
