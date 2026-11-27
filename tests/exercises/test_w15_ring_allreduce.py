import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def test_ring():
    rng = np.random.default_rng(0)
    for N in (2, 3, 5, 8):
        bufs = [rng.normal(size=1000) for _ in range(N)]
        res, sent = m.ring_allreduce(bufs)
        for r in res:
            np.testing.assert_allclose(r, sum(bufs))
        expected = 2 * (N - 1) / N * 1000 * 8
        assert all(abs(s - expected) <= 2 * (N - 1) * 8 for s in sent)
