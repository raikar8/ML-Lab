import random
import time

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    assert m.window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert m.window_max([4], 1) == [4] and m.window_max([1, 2, 3], 3) == [3]


def test_random_against_brute_force():
    rng = random.Random(33)
    for _ in range(500):
        a = [rng.randint(-5, 5) for _ in range(rng.randint(1, 30))]
        k = rng.randint(1, len(a))
        assert m.window_max(a, k) == [max(a[i:i + k]) for i in range(len(a) - k + 1)]
        ph = m.PeakHold(k)
        assert [ph.push(x) for x in a] == [max(a[max(0, i - k + 1):i + 1]) for i in range(len(a))]


def test_linear_big_k():
    a = list(range(300_000, 0, -1))
    t0 = time.perf_counter()
    out = m.window_max(a, 50_000)
    assert out[0] == 300_000 and time.perf_counter() - t0 < 3
