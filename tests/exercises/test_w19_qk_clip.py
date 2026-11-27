import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_qk_clip():
    torch.manual_seed(0)
    Wq, Wk, x = torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(10, 16)
    Wq[1] *= 10
    before = m.max_logits(Wq, Wk, x)
    tau = float(before.sort().values[2])
    q0 = Wq.clone()
    f = m.qk_clip_(Wq, Wk, x, tau)
    after = m.max_logits(Wq, Wk, x)
    assert (after <= tau + 1e-4).all()
    clipped = before > tau
    torch.testing.assert_close(after[clipped], torch.full((int(clipped.sum()),), tau), atol=1e-4, rtol=1e-4)
    assert torch.equal(Wq[~clipped], q0[~clipped]) and (f[~clipped] == 1).all()
