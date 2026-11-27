import torch
import torch.nn as nn
from helpers import key_of, load

m = load(key_of(__file__))


def test_layernorm_rmsnorm():
    x = torch.randn(3, 5, 16) * 3 + 1
    ln = m.LayerNorm(16)
    with torch.no_grad():
        ln.weight.uniform_(); ln.bias.uniform_()
    ref = nn.LayerNorm(16); ref.load_state_dict(ln.state_dict())
    torch.testing.assert_close(ln(x), ref(x))
    rn = m.RMSNorm(16)
    ref = nn.RMSNorm(16, eps=1e-6)
    torch.testing.assert_close(rn(x), ref(x))
    assert rn(x.bfloat16()).dtype == torch.bfloat16


def test_batchnorm_train_eval():
    torch.manual_seed(0)
    bn, ref = m.BatchNorm1d(4), nn.BatchNorm1d(4)
    assert set(dict(bn.named_buffers())) >= {"running_mean", "running_var"}
    for _ in range(5):
        x = torch.randn(8, 4) * 2 + 3
        torch.testing.assert_close(bn(x), ref(x))
    torch.testing.assert_close(bn.running_var, ref.running_var)
    bn.eval(); ref.eval()
    x = torch.randn(8, 4)
    torch.testing.assert_close(bn(x), ref(x))


def test_fold():
    lin, bn = nn.Linear(6, 4), nn.BatchNorm1d(4)
    bn.train()
    for _ in range(3):
        bn(lin(torch.randn(16, 6)))
    bn.eval()
    x = torch.randn(5, 6)
    with torch.no_grad():
        torch.testing.assert_close(m.fold_bn_into_linear(lin, bn)(x), bn(lin(x)), atol=1e-5, rtol=1e-5)
