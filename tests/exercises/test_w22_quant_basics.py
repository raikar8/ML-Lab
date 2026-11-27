import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def test_params_and_roundtrip():
    s, z = m.qparams_asym(0.5, 6.0)
    assert z == 0 and abs(s - 6 / 255) < 1e-12  # range widened to include 0
    s, z = m.qparams_asym(-1.0, 3.0)
    q = m.quantize(np.array([0.0]), s, z, 0, 255)
    assert m.dequantize(q, s, z)[0] == 0.0  # zero exactly representable
    assert abs(m.qparams_sym(1.27) - 0.01) < 1e-12


def test_sqnr_and_clipping():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100000)
    s8, s4 = m.qparams_sym(np.abs(x).max(), 8), m.qparams_sym(np.abs(x).max(), 4)
    d8 = m.sqnr_db(x, m.dequantize(m.quantize(x, s8, 0, -127, 127), s8, 0))
    d4 = m.sqnr_db(x, m.dequantize(m.quantize(x, s4, 0, -7, 7), s4, 0))
    assert 20 < d8 - d4 < 28
    c = m.best_clip_mse(x, bits=4)
    assert 1.5 < c < 3.5 < np.abs(x).max()
