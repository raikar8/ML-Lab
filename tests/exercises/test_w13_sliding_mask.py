import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_mask():
    mk = m.sliding_window_mask(6, 2, n_sink=1)
    assert mk[5].tolist() == [True, False, False, False, True, True]
    assert mk[1].tolist() == [True, True, False, False, False, False]
    assert not mk.triu(1).any()


def test_rolling_cache():
    c = m.RollingKVCache(window=3, n_sink=2)
    for t in range(10):
        x = torch.full((1, 1, 1, 2), float(t))
        c.append(x, x)
    k, v = c.get()
    assert k[0, 0, :, 0].tolist() == [0, 1, 7, 8, 9] and c.positions() == [0, 1, 7, 8, 9]
