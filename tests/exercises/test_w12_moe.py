import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_matches_dense_reference():
    torch.manual_seed(0)
    moe = m.MoE(8, 16, 4, 2, n_shared=1).double()
    x = torch.randn(2, 5, 8, dtype=torch.double)
    y, aux = moe(x)
    xf = x.reshape(-1, 8)
    p = torch.softmax(moe.router(xf), -1)
    w, idx = p.topk(2, -1); w = w / w.sum(-1, keepdim=True)
    ref = torch.stack([sum(w[n, j] * moe.experts[int(idx[n, j])](xf[n]) for j in range(2)) for n in range(10)])
    ref = ref + moe.shared[0](xf)
    torch.testing.assert_close(y.reshape(-1, 8), ref)
    assert moe.last_load.sum() == 20 and aux.ndim == 0
    y.sum().backward()
    assert moe.router.weight.grad is not None


def test_aux_loss_value():
    torch.manual_seed(1)
    moe = m.MoE(4, 8, 4, 1)
    with torch.no_grad():
        moe.router.weight.zero_()
    _, aux = moe(torch.randn(1, 8, 4))
    assert abs(aux.item() - 1.0) < 1e-6  # uniform probs -> E * sum(f_i * 1/E) = 1
