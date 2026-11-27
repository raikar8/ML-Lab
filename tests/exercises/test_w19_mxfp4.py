import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_e2m1_and_mx():
    x = torch.tensor([0.2, 0.3, 0.74, 1.2, 2.6, -3.4, 5.1, 9.0])
    assert m.quant_e2m1(x).tolist() == [0.0, 0.5, 0.5, 1.0, 3.0, -3.0, 6.0, 6.0]
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    q = m.mxfp4(w)
    blocks = (q.reshape(64, -1, 32) / w.reshape(64, -1, 32).abs().amax(-1, keepdim=True).log2().floor().exp2().mul(0.25))
    assert torch.isin(blocks.abs().round(decimals=4), m.FP4_GRID).all()
    e_mx = (w - q).norm() / w.norm()
    e_i4 = (w - m.int4_group(w)).norm() / w.norm()
    assert 0.05 < e_mx < 0.2 and 0.05 < e_i4 < 0.2
    assert m.bits_per_value("mxfp4") == 4.25
