import random
import time

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    tm = m.TimeMap()
    tm.set("foo", "bar", 1)
    assert tm.get("foo", 1) == "bar" and tm.get("foo", 3) == "bar" and tm.get("foo", 0) is None
    tm.set("foo", "bar2", 4)
    assert tm.get("foo", 4) == "bar2" and tm.get("foo", 3) == "bar" and tm.get("nope", 9) is None
    assert tm.history("foo", 0, 10) == [(1, "bar"), (4, "bar2")] and tm.history("foo", 2, 3) == []


def test_random_against_brute_force():
    rng = random.Random(53)
    tm, ref = m.TimeMap(), {}
    t = {k: 0 for k in "abc"}
    for _ in range(3000):
        k = rng.choice("abc")
        if rng.random() < 0.4:
            t[k] += rng.randint(1, 3)
            v = rng.randint(0, 99)
            tm.set(k, v, t[k]); ref.setdefault(k, []).append((t[k], v))
        else:
            q = rng.randint(0, t[k] + 2)
            older = [v for ts, v in ref.get(k, []) if ts <= q]
            assert tm.get(k, q) == (older[-1] if older else None)
            a = rng.randint(0, q)
            assert tm.history(k, a, q) == [(ts, v) for ts, v in ref.get(k, []) if a <= ts <= q]


def test_log_time_get():
    tm = m.TimeMap()
    for i in range(200_000):
        tm.set("k", i, i * 2)
    t0 = time.perf_counter()
    for i in range(20_000):
        assert tm.get("k", 2 * i + 1) == i
    assert time.perf_counter() - t0 < 1.5
