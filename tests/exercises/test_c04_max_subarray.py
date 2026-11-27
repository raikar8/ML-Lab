import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(nums):
    best = None
    for a in range(len(nums)):
        for b in range(a + 1, len(nums) + 1):
            cand = (-sum(nums[a:b]), a, b - a)
            if best is None or cand < best:
                best = cand
    s, a, L = best
    return (-s, a, a + L)


def test_examples():
    assert m.max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == (6, 3, 7)
    assert m.max_subarray([-3, -1, -2]) == (-1, 1, 2)
    assert m.max_subarray([5]) == (5, 0, 1)
    assert m.max_subarray([0, 0, 0]) == (0, 0, 1)


def test_random_against_brute_force():
    rng = random.Random(3)
    for _ in range(1500):
        nums = [rng.randint(-6, 6) for _ in range(rng.randint(1, 12))]
        assert m.max_subarray(nums) == brute(nums), nums


def test_linear():
    nums = [(-1) ** i * (i % 7) for i in range(300_000)]
    s, a, b = m.max_subarray(nums)
    assert s == sum(nums[a:b])
