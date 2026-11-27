import torch
import torch.nn as nn
from helpers import key_of, load

m = load(key_of(__file__))


def test_shapes_params_no_bias():
    blk = m.DecoderBlock(64, 8, 2, 172)
    assert blk(torch.randn(2, 10, 64)).shape == (2, 10, 64)
    assert sum(p.numel() for p in blk.parameters()) == m.param_count(64, 8, 2, 172)
    assert all(mod.bias is None for mod in blk.modules() if isinstance(mod, nn.Linear))
    assert blk.wk.out_features == 16


def test_causal_and_gqa_equals_mha_when_heads_equal():
    torch.manual_seed(0)
    blk = m.DecoderBlock(32, 4, 4, 64).double()
    x = torch.randn(1, 7, 32, dtype=torch.double)
    y = blk(x)
    x2 = x.clone(); x2[:, 5:] += 3
    torch.testing.assert_close(blk(x2)[:, :5], y[:, :5])
    blk2 = m.DecoderBlock(32, 4, 1, 64).double()
    sd = blk2.state_dict()
    for name in ("wk.weight", "wv.weight"):
        sd[name] = blk.state_dict()[name][:8]
    for name in sd:
        if name not in ("wk.weight", "wv.weight"):
            sd[name] = blk.state_dict()[name]
    blk2.load_state_dict(sd)
    blk3 = m.DecoderBlock(32, 4, 4, 64).double()
    sd3 = dict(blk.state_dict())
    sd3["wk.weight"] = sd["wk.weight"].repeat(4, 1); sd3["wv.weight"] = sd["wv.weight"].repeat(4, 1)
    blk3.load_state_dict(sd3)
    torch.testing.assert_close(blk2(x), blk3(x))
