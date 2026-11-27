import inspect
import math

import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_frame_is_view():
    x = torch.arange(20.0).reshape(2, 10)
    f = m.frame_signal(x, 4, 2)
    assert f.shape == (2, 4, 4)
    assert f.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert torch.equal(f[1, 2], torch.tensor([14.0, 15, 16, 17]))


def test_pairwise_and_scores():
    a, b = torch.randn(2, 5, 3, dtype=torch.double), torch.randn(2, 4, 3, dtype=torch.double)
    torch.testing.assert_close(m.pairwise_l2(a, b), torch.cdist(a, b))
    q, k = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 6, 8)
    torch.testing.assert_close(m.attention_scores(q, k), q @ k.transpose(-1, -2) / math.sqrt(8))
    assert "einsum" in inspect.getsource(m.attention_scores)


def test_masks():
    cm = m.causal_mask(3)
    assert cm.dtype == torch.bool and cm.tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    pm = m.padding_mask(torch.tensor([1, 3]), 3)
    assert pm.tolist() == [[True, False, False], [True, True, True]]
