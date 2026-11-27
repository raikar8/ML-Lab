import torch
import torch.nn.functional as F
from helpers import key_of, load

m = load(key_of(__file__))


def test_sdpa(monkeypatch):
    q, k, v = (torch.randn(2, 3, 5, 8, dtype=torch.double) for _ in range(3))
    mask = torch.rand(2, 1, 5, 5) > 0.3
    mask[..., 0] = True
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    monkeypatch.setattr(F, "scaled_dot_product_attention", None)
    torch.testing.assert_close(m.sdpa(q, k, v, mask), ref)


def test_mha_matches_reference_and_masks():
    torch.manual_seed(0)
    mha = m.MultiHeadAttention(16, 4).double()
    x = torch.randn(2, 6, 16, dtype=torch.double)
    q, k, v = mha.qkv(x).view(2, 6, 3, 4, 4).permute(2, 0, 3, 1, 4)
    ref = mha.out(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 6, 16))
    torch.testing.assert_close(mha(x, causal=True), ref)
    # causality: changing future tokens doesn't change earlier outputs
    x2 = x.clone(); x2[:, 4:] += 1
    torch.testing.assert_close(mha(x2, causal=True)[:, :4], mha(x, causal=True)[:, :4])
    # padding: padded keys are ignored
    kpm = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    x3 = x.clone(); x3[1, 4:] = 100
    torch.testing.assert_close(mha(x3, key_padding_mask=kpm)[1, :4], mha(x, key_padding_mask=kpm)[1, :4])
