import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_gptq_beats_rtn():
    torch.manual_seed(0)
    W = torch.randn(32, 64, dtype=torch.double)
    mix = torch.randn(64, 64, dtype=torch.double) * 0.3 + torch.eye(64, dtype=torch.double)
    X = torch.randn(512, 64, dtype=torch.double) @ mix
    e_rtn = m.output_error(W, m.rtn(W), X)
    e_gptq = m.output_error(W, m.gptq(W, X), X)
    assert e_gptq < 0.7 * e_rtn
    Q = m.gptq(W, X)
    s = W.abs().amax(1, keepdim=True) / 7
    assert torch.allclose(Q / s, torch.round(Q / s), atol=1e-6)
