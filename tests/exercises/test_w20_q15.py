import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def test_q15_ops():
    assert m.to_q15(0.5) == 16384 and m.to_q15(1.0) == 32767 and m.to_q15(-1.0) == -32768
    assert m.q15_mul(m.to_q15(0.5), m.to_q15(-0.5)) == -8192
    assert m.q15_mul(-32768, -32768) == 32767
    assert m.q15_add(30000, 10000) == 32767 and m.q15_add(-30000, -10000) == -32768
    assert abs(m.from_q15(m.q15_mul(m.to_q15(0.3), m.to_q15(0.7))) - 0.21) < 1e-4


def test_fir_matches_float():
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, 200)
    h = np.hanning(15); h /= h.sum() * 1.05
    y = np.convolve(x, h)[:200]
    yq = m.fir_q15([m.to_q15(v) for v in x], [m.to_q15(v) for v in h])
    assert np.abs(yq / 32768 - y).max() < 5e-4
