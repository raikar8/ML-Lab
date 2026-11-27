import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(a, k):
    hits = [(j - i) for i in range(len(a)) for j in range(i + 1, len(a) + 1) if sum(a[i:j]) == k]
    return len(hits), max(hits, default=0)


def test_examples():
    assert m.count_subarrays([1, 1, 1], 2) == 2
    assert m.count_subarrays([1, -1, 1, -1], 0) == 4
    assert m.longest_subarray([1, -1, 5, -2, 3], 3) == 4
    assert m.count_subarrays([], 0) == 0 and m.longest_subarray([2], 3) == 0


def test_random_against_brute_force():
    rng = random.Random(19)
    for _ in range(500):
        a = [rng.randint(-4, 4) for _ in range(rng.randint(0, 16))]
        k = rng.randint(-5, 5)
        assert (m.count_subarrays(a, k), m.longest_subarray(a, k)) == brute(a, k), (a, k)


def test_linear():
    a = [1, -1] * 100_000
    assert m.count_subarrays(a, 0) == 100_000 ** 2
