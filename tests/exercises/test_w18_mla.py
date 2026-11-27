import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_mla():
    torch.manual_seed(0)
    att = m.MLA(32, 4, 8, 12).double()
    x = torch.randn(2, 7, 32, dtype=torch.double)
    full, c = att(x)
    assert c.shape == (2, 7, 12)
    torch.testing.assert_close(att.forward_absorbed(x)[0], full)
    o1, cache = att(x[:, :5])
    o2, cache = att.forward_absorbed(x[:, 5:], cache)
    torch.testing.assert_close(torch.cat([o1, o2], 1), full)
    assert att.cache_values_per_token(61) == 12 * 61
