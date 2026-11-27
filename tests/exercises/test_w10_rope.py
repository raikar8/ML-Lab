import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_relative_property():
    d = 16
    q, k = torch.randn(d, dtype=torch.double), torch.randn(d, dtype=torch.double)
    cos, sin = m.rope_cache(64, d)

    def score(i, j):
        return float(m.apply_rope(q, cos[i], sin[i]) @ m.apply_rope(k, cos[j], sin[j]))

    assert abs(score(10, 3) - score(40, 33)) < 1e-9
    assert abs(score(5, 5) - float(q @ k)) < 1e-9
    assert abs(score(10, 3) - score(10, 4)) > 1e-6


def test_norm_offset_partial():
    x = torch.randn(2, 8, 32, dtype=torch.double)
    cos, sin = m.rope_cache(8, 32)
    torch.testing.assert_close(m.apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))
    c2, s2 = m.rope_cache(4, 32, offset=4)
    torch.testing.assert_close(c2, cos[4:])
    cp, sp = m.rope_cache(8, 8)
    y = m.partial_rope(x, cp, sp, 0.25)
    torch.testing.assert_close(y[..., 8:], x[..., 8:])
    torch.testing.assert_close(y[..., :8], m.apply_rope(x[..., :8], cp, sp))
    assert m.rotate_half(torch.tensor([1.0, 2, 3, 4])).tolist() == [-3, -4, 1, 2]
