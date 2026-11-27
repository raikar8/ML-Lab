import pytest
import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_warmup_cosine():
    lr = [m.warmup_cosine(s, 100, 10, 1.0, 0.1) for s in range(100)]
    assert lr[0] == pytest.approx(0.1) and lr[9] == pytest.approx(1.0)
    assert lr[-1] == pytest.approx(0.1) and all(a >= b for a, b in zip(lr[9:], lr[10:]))
    assert m.warmup_cosine(54, 100, 10, 1.0, 0.1) == pytest.approx(0.55, abs=0.02)


def test_wsd():
    lr = [m.wsd(s, 100, 5, 0.2, 2.0, 0.0) for s in range(100)]
    assert lr[4] == 2.0 and lr[50] == 2.0 and lr[79] == 2.0
    assert lr[-1] == pytest.approx(0.0) and lr[90] < lr[85] < 2.0


def test_lambda_with_torch():
    w = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([w], lr=3.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, m.as_lambda(m.warmup_cosine, 3.0, total=20, warmup=4))
    seen = []
    for _ in range(20):
        seen.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
    assert seen[3] == pytest.approx(3.0) and seen[0] == pytest.approx(0.75)
