import copy

import torch
import torch.nn as nn
from helpers import key_of, load

m = load(key_of(__file__))


def data(n=256):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, 2, generator=g)
    y = (X[:, 0] * X[:, 1] > 0).long()
    return X, y


def test_learns_and_eval_mode():
    torch.manual_seed(0)
    X, y = data()
    net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2))
    dl = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=32, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    for _ in range(40):
        losses = m.train_epoch(net, dl, opt, nn.functional.cross_entropy, clip=1.0)
    assert len(losses) == 8
    net.train()
    loss, acc = m.evaluate(net, dl, nn.functional.cross_entropy)
    assert acc > 0.93 and net.training


def test_accumulation_equals_big_batch():
    torch.manual_seed(1)
    X, y = data(32)
    net = nn.Linear(2, 2)
    net2 = copy.deepcopy(net)
    o1, o2 = torch.optim.SGD(net.parameters(), 0.1), torch.optim.SGD(net2.parameters(), 0.1)
    small = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=8)
    big = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=32)
    m.train_epoch(net, small, o1, nn.functional.cross_entropy, accum_steps=4)
    m.train_epoch(net2, big, o2, nn.functional.cross_entropy)
    torch.testing.assert_close(net.weight, net2.weight)


def test_remainder_step():
    X, y = data(24)
    net = nn.Linear(2, 2); w0 = net.weight.detach().clone()
    dl = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=8)  # 3 batches, accum 2 -> remainder
    calls = []
    opt = torch.optim.SGD(net.parameters(), 0.1)
    orig = opt.step
    opt.step = lambda *a, **k: (calls.append(1), orig(*a, **k))
    m.train_epoch(net, dl, opt, nn.functional.cross_entropy, accum_steps=2)
    assert len(calls) == 2 and not torch.equal(w0, net.weight)
