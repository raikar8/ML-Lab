import torch
import torch.nn.functional as F
from helpers import key_of, load

m = load(key_of(__file__))


def test_indexer_and_dense_equivalence():
    torch.manual_seed(0)
    B, H, T, D = 2, 2, 10, 8
    qi, ki, w = torch.randn(B, 3, T, 4), torch.randn(B, T, 4), torch.rand(B, 3, T)
    I = m.indexer_scores(qi, ki, w)
    assert I.shape == (B, T, T) and torch.isinf(I[0, 0, 1]) and I[0, 5, 3] >= 0
    q, k, v = (torch.randn(B, H, T, D, dtype=torch.double) for _ in range(3))
    dense = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(m.topk_sparse_attention(q, k, v, I, topk=T), dense)


def test_selects_only_topk():
    q = torch.zeros(1, 1, 4, 2, dtype=torch.double)
    k = torch.zeros(1, 1, 4, 2, dtype=torch.double)
    v = torch.eye(4, dtype=torch.double)[None, None]
    scores = torch.tensor([[[0.0, -1e9, -1e9, -1e9], [0, 1, -1e9, -1e9], [5, 0, 1, -1e9], [0, 9, 1, 8]]],
                          dtype=torch.double).masked_fill(torch.ones(4, 4).triu(1).bool(), float("-inf"))
    out = m.topk_sparse_attention(q, k, v, scores, topk=2)[0, 0]
    torch.testing.assert_close(out[3], torch.tensor([0, 0.5, 0, 0.5], dtype=torch.double))
    torch.testing.assert_close(out[2], torch.tensor([0.5, 0, 0.5, 0], dtype=torch.double))
