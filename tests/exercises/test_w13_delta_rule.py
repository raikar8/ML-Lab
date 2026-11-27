import torch
import torch.nn.functional as F
from helpers import key_of, load

m = load(key_of(__file__))


def test_matches_closed_form_step():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 1, 3, 4, dtype=torch.double), torch.randn(1, 1, 3, 4, dtype=torch.double), torch.randn(1, 1, 3, 2, dtype=torch.double)
    beta, alpha = torch.rand(1, 1, 3, dtype=torch.double), torch.rand(1, 1, 3, dtype=torch.double)
    kn, qn = F.normalize(k, dim=-1)[0, 0], F.normalize(q, dim=-1)[0, 0]
    S = torch.zeros(2, 4, dtype=torch.double)
    ref = []
    for t in range(3):
        S = alpha[0, 0, t] * S @ (torch.eye(4, dtype=torch.double) - beta[0, 0, t] * torch.outer(kn[t], kn[t])) + beta[0, 0, t] * torch.outer(v[0, 0, t], kn[t])
        ref.append(S @ qn[t])
    torch.testing.assert_close(m.gated_delta_rule(q, k, v, beta, alpha)[0, 0], torch.stack(ref))


def test_overwrite_recall_beats_additive():
    torch.manual_seed(1)
    D = 16
    keys = torch.linalg.qr(torch.randn(D, D))[0][:4]          # 4 orthonormal keys
    seq_k = torch.cat([keys, keys])                             # write each key twice: old value then new value
    old, new = torch.randn(4, D), torch.randn(4, D)
    seq_v = torch.cat([old, new])
    k = seq_k[None, None]; v = seq_v[None, None]
    q = keys[None, None]
    beta = torch.ones(1, 1, 8)
    # query after the sequence: append the queries as zero-beta steps
    kq = torch.cat([k, q], 2); vq = torch.cat([v, torch.zeros(1, 1, 4, D)], 2)
    bq = torch.cat([beta, torch.zeros(1, 1, 4)], 2)
    out = m.gated_delta_rule(torch.cat([k, q], 2), kq, vq, bq)[0, 0, 8:]
    torch.testing.assert_close(out, new, atol=1e-4, rtol=1e-4)
    add = m.additive_linear_attention(torch.cat([k, q], 2), kq, vq)[0, 0, 8:]
    assert (add - new).norm() > 0.5 * new.norm()
