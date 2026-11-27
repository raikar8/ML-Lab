import numpy as np
import torch
import torch.nn.functional as F
from helpers import key_of, load

m = load(key_of(__file__))


def test_stability_and_values():
    z = np.array([[1000.0, 1000.0, -1000.0], [1.0, 2.0, 3.0]])
    assert np.all(np.isfinite(m.logsumexp(z)))
    np.testing.assert_allclose(m.softmax(z)[0], [0.5, 0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(m.softmax(z).sum(1), 1.0)
    np.testing.assert_allclose(m.log_softmax(z)[1], F.log_softmax(torch.tensor(z[1]), 0).numpy(), atol=1e-12)


def test_cross_entropy_grad_matches_torch():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5, 4)) * 3
    y = rng.integers(0, 4, size=5)
    loss, dz = m.cross_entropy(z, y)
    zt = torch.tensor(z, requires_grad=True)
    lt = F.cross_entropy(zt, torch.tensor(y))
    lt.backward()
    assert abs(loss - lt.item()) < 1e-10
    np.testing.assert_allclose(dz, zt.grad.numpy(), atol=1e-12)
