import numpy as np
import torch
import torch.nn.functional as F
from helpers import key_of, load

m = load(key_of(__file__))


def test_grads_match_autograd():
    rng = np.random.default_rng(0)
    p = m.init_params(5, 7, 3, rng)
    X, y = rng.normal(size=(11, 5)), rng.integers(0, 3, 11)
    loss, g = m.loss_and_grads(p, X, y)
    tp = {k: torch.tensor(v, requires_grad=True) for k, v in p.items()}
    lt = F.cross_entropy(torch.relu(torch.tensor(X) @ tp["W1"] + tp["b1"]) @ tp["W2"] + tp["b2"], torch.tensor(y))
    lt.backward()
    assert abs(loss - lt.item()) < 1e-10
    for k in p:
        assert g[k].shape == p[k].shape
        np.testing.assert_allclose(g[k], tp[k].grad.numpy(), atol=1e-10)


def test_trains_on_spiral():
    rng = np.random.default_rng(1)
    n = 100
    t = np.linspace(0, 4, n)
    X = np.concatenate([np.c_[t * np.cos(t + 2.1 * k), t * np.sin(t + 2.1 * k)] for k in range(3)]) / 4
    y = np.repeat(np.arange(3), n)
    p = m.train(X, y, 64, 3, lr=1.0, steps=2000, rng=rng)
    assert (m.forward(p, X)[0].argmax(1) == y).mean() > 0.9
