import pytest
from helpers import key_of, load

m = load(key_of(__file__))


def test_estimates():
    assert m.dense_params(96, 12288, 50257) / 1e9 == pytest.approx(174.6, rel=0.01)
    assert m.train_flops(37e9, 14.8e12) == pytest.approx(3.29e24, rel=0.01)
    assert m.chinchilla_tokens(70e9) == 1.4e12
    assert m.weight_bytes(7e9, 4) == 3.5e9
    assert m.decode_tokens_per_s(7e9, 4, 70) == pytest.approx(20.0)
    assert m.decode_tokens_per_s(7e9, 4, 70, kv_bytes_per_token=131072, context=4096) < 20
    assert m.prefill_tokens_per_s(3e9, 45, 0.3) == pytest.approx(2250)
    assert m.train_memory_gb(7e9) == 112
