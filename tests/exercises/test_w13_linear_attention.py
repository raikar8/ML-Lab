import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_forms_agree_and_causal():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 13, 4, dtype=torch.double) for _ in range(3))
    a = m.linear_attention_parallel(q, k, v)
    torch.testing.assert_close(m.linear_attention_recurrent(q, k, v), a)
    for c in (1, 4, 5, 13):
        torch.testing.assert_close(m.linear_attention_chunked(q, k, v, c), a)
    v2 = v.clone(); v2[:, :, 8:] += 1
    torch.testing.assert_close(m.linear_attention_parallel(q, k, v2)[:, :, :8], a[:, :, :8])
