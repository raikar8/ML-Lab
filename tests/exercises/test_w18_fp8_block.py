import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_fp8_scaling():
    torch.manual_seed(0)
    x = torch.randn(64, 512) * 1e-3
    x[:, 7] *= 1e4
    pt, tile = m.rel_error(x, m.fp8_per_tensor(x)), m.rel_error(x, m.fp8_tile_1x128(x))
    assert tile < pt and m.rel_error(x[:, 128:], m.fp8_tile_1x128(x)[:, 128:]) < 0.05
    w = torch.randn(256, 256); w[:128, :128] *= 1e5
    assert m.rel_error(w[128:], m.fp8_block_128(w)[128:]) < 0.05 < m.rel_error(w[128:], m.fp8_per_tensor(w)[128:])
