import torch
import torch.nn.functional as F
from helpers import key_of, load

m = load(key_of(__file__))


def test_matches_sdpa():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 37, 8, dtype=torch.double) * 3 for _ in range(3))
    for causal in (False, True):
        o, lse = m.flash_attention(q, k, v, 8, 16, causal)
        torch.testing.assert_close(o, F.scaled_dot_product_attention(q, k, v, is_causal=causal))
    s = q @ k.transpose(-1, -2) / 8 ** 0.5
    torch.testing.assert_close(m.flash_attention(q, k, v)[1], torch.logsumexp(s, -1))


def test_never_builds_full_matrix(monkeypatch):
    q = torch.randn(1, 1, 64, 4)
    orig = torch.Tensor.__matmul__
    def guarded(a, b):
        out = orig(a, b)
        assert not (out.shape[-1] == 64 and out.shape[-2] == 64), "full T x T matrix built"
        return out
    monkeypatch.setattr(torch.Tensor, "__matmul__", guarded)
    m.flash_attention(q, q, q, 16, 16)


def test_combine_partials():
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 2, 1, 8, dtype=torch.double),
               torch.randn(1, 2, 20, 8, dtype=torch.double), torch.randn(1, 2, 20, 8, dtype=torch.double))
    def part(kk, vv):
        s = q @ kk.transpose(-1, -2) / 8 ** 0.5
        return torch.softmax(s, -1) @ vv, torch.logsumexp(s, -1)
    o, _ = m.combine_partials(*part(k[:, :, :7], v[:, :, :7]), *part(k[:, :, 7:], v[:, :, 7:]))
    torch.testing.assert_close(o, F.scaled_dot_product_attention(q, k, v))
