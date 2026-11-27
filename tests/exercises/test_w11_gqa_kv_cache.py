import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_incremental_equals_full():
    torch.manual_seed(0)
    att = m.GQAttention(32, 8, 2).double()
    x = torch.randn(2, 9, 32, dtype=torch.double)
    full, (k, v) = att(x)
    assert k.shape == (2, 2, 9, 4)
    out, cache = att(x[:, :4])
    outs = [out]
    for t in range(4, 9):
        o, cache = att(x[:, t:t + 1], cache)
        outs.append(o)
    torch.testing.assert_close(torch.cat(outs, 1), full)
    o2, _ = att(x[:, 6:], att(x[:, :6])[1])  # multi-token chunk after cache
    torch.testing.assert_close(o2, full[:, 6:])


def test_bytes():
    assert m.kv_cache_bytes(32, 32, 128, 4096) == 2 * 1024 ** 3
    assert m.kv_cache_bytes(32, 8, 128, 4096) == 512 * 1024 ** 2
