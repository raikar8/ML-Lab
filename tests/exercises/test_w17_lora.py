import torch
import torch.nn as nn
from helpers import key_of, load

m = load(key_of(__file__))


def test_lora():
    torch.manual_seed(0)
    base = nn.Linear(16, 8).double()
    x = torch.randn(4, 16, dtype=torch.double)
    ref = base(x).detach()
    lo = m.LoRALinear(base, r=4, alpha=8).double()
    torch.testing.assert_close(lo(x), ref)  # B = 0 at init
    assert not base.weight.requires_grad and len(m.lora_parameters(nn.Sequential(lo))) == 2
    opt = torch.optim.SGD(m.lora_parameters(lo), 0.002)
    for _ in range(5):
        opt.zero_grad(); ((lo(x) - 1) ** 2).sum().backward(); opt.step()
    y = lo(x).detach()
    w0 = base.weight.clone()
    lo.merge()
    torch.testing.assert_close(lo(x), y)
    lo.unmerge()
    torch.testing.assert_close(base.weight, w0)
    assert sum(p.numel() for p in m.lora_parameters(lo)) == 4 * (16 + 8)
