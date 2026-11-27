import random

from helpers import key_of, load

m = load(key_of(__file__))


def test_token_bucket():
    tb = m.TokenBucket(rate=2.0, capacity=3)
    assert [tb.allow(0.0) for _ in range(4)] == [True, True, True, False]
    assert tb.allow(0.5) and not tb.allow(0.5)        # 1 token refilled in 0.5 s
    assert [tb.allow(10.0) for _ in range(4)] == [True, True, True, False]   # capped at capacity


def test_token_bucket_long_run_rate():
    tb = m.TokenBucket(rate=5.0, capacity=1)
    accepted = sum(tb.allow(t / 100) for t in range(10_000))   # 100 s, requests every 10 ms
    assert abs(accepted - 500) <= 2


def test_sliding_window():
    sw = m.SlidingWindowLimiter(limit=2, window=10)
    assert sw.allow("a", 0) and sw.allow("a", 1) and not sw.allow("a", 5)
    assert sw.allow("b", 5)                            # keys are independent
    assert sw.allow("a", 10) and not sw.allow("a", 10.5) and sw.allow("a", 11)


def test_sliding_window_random_against_brute_force():
    rng = random.Random(42)
    sw = m.SlidingWindowLimiter(limit=3, window=5)
    acc = {"x": [], "y": []}
    t = 0.0
    for _ in range(3000):
        t += rng.choice([0, 0.5, 1, 2])
        k = rng.choice("xy")
        exp = sum(1 for s in acc[k] if t - 5 < s <= t) < 3
        got = sw.allow(k, t)
        assert got == exp
        if got:
            acc[k].append(t)
