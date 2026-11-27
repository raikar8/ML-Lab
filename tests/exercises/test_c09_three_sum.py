import itertools
import random
import time

from helpers import key_of, load

m = load(key_of(__file__))


def brute(nums, t=0):
    return sorted({tuple(sorted(c)) for c in itertools.combinations(nums, 3) if sum(c) == t})


def test_examples():
    assert m.three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
    assert m.three_sum([0, 0, 0, 0]) == [[0, 0, 0]]
    assert m.three_sum([1, 2]) == []
    assert m.three_sum([1, 2, 3, 4], target=9) == [[2, 3, 4]]


def test_random_against_brute_force():
    rng = random.Random(9)
    for _ in range(300):
        nums = [rng.randint(-6, 6) for _ in range(rng.randint(0, 14))]
        t = rng.randint(-3, 3)
        assert [tuple(x) for x in m.three_sum(nums, t)] == brute(nums, t)


def test_quadratic():
    rng = random.Random(0)
    nums = [rng.randint(-10**6, 10**6) for _ in range(1500)]
    t0 = time.perf_counter()
    m.three_sum(nums)
    assert time.perf_counter() - t0 < 5
