import functools
import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute_speed(piles, h):
    v = 1
    while sum(-(-p // v) for p in piles) > h:
        v += 1
    return v


def brute_split(c, k):
    @functools.lru_cache(None)
    def go(i, k):
        if i == len(c):
            return 0
        if k == 0:
            return float("inf")
        best, s = float("inf"), 0
        for j in range(i, len(c)):
            s += c[j]
            best = min(best, max(s, go(j + 1, k - 1)))
        return best
    return go(0, k)


def test_examples():
    assert m.min_eating_speed([3, 6, 7, 11], 8) == 4
    assert m.min_eating_speed([30, 11, 23, 4, 20], 5) == 30
    assert m.split_min_max([7, 2, 5, 10, 8], 2) == 18 and m.split_min_max([1, 2, 3], 5) == 3


def test_random():
    rng = random.Random(45)
    for _ in range(300):
        piles = [rng.randint(1, 30) for _ in range(rng.randint(1, 6))]
        h = rng.randint(len(piles), len(piles) + 20)
        assert m.min_eating_speed(piles, h) == brute_speed(piles, h)
        c = [rng.randint(1, 9) for _ in range(rng.randint(1, 8))]
        k = rng.randint(1, 5)
        assert m.split_min_max(c, k) == brute_split(tuple(c), k)


def test_big_values():
    assert m.min_eating_speed([10**9] * 1000, 10**6) == 10**6
