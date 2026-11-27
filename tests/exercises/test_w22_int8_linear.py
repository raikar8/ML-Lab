import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))


def test_multiplier():
    for M in (0.3, 0.000123, 0.9999):
        M0, sh = m.quantize_multiplier(M)
        assert 2 ** 30 <= M0 < 2 ** 31 and abs(M0 / 2 ** 31 * 2 ** -sh - M) / M < 1e-8


def test_int8_linear_within_one_lsb():
    rng = np.random.default_rng(0)
    N, K, C = 16, 64, 8
    x = rng.uniform(-0.5, 2.0, (N, K)); W = rng.normal(0, 0.1, (C, K)); b = rng.normal(0, 0.1, C)
    s_x = (x.max() - x.min()) / 255; zp_x = int(round(-128 - x.min() / s_x))
    q_x = np.clip(np.round(x / s_x) + zp_x, -128, 127).astype(np.int64)
    s_w = np.abs(W).max(1) / 127; q_w = np.round(W / s_w[:, None]).astype(np.int64)
    y = x @ W.T + b
    s_y = (y.max() - y.min()) / 255; zp_y = int(round(-128 - y.min() / s_y))
    b_q = np.round(b / (s_x * s_w)).astype(np.int64)
    Ms = [m.quantize_multiplier(s_x * s_w[c] / s_y) for c in range(C)]
    q_y = m.int8_linear(q_x, zp_x, q_w, b_q, [a for a, _ in Ms], [s for _, s in Ms], zp_y)
    xq = s_x * (q_x - zp_x); wq = s_w[:, None] * q_w
    ref = np.clip(np.round((xq @ wq.T + s_x * s_w * b_q) / s_y) + zp_y, -128, 127)
    assert np.abs(q_y - ref).max() <= 1
    assert np.abs(s_y * (q_y - zp_y) - y).max() < 3 * s_y
