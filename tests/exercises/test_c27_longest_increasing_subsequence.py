import random
import time

from helpers import key_of, load

m = load(key_of(__file__))


def brute_len(a):
    best = [1] * len(a)
    for i in range(len(a)):
        for j in range(i):
            if a[j] < a[i]:
                best[i] = max(best[i], best[j] + 1)
    return max(best, default=0)


def is_subseq(sub, a):
    it = iter(a)
    return all(any(x == y for y in it) for x in sub)


def test_examples():
    assert m.lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert m.lis_length([]) == 0 and m.lis([]) == [] and m.lis_length([7, 7, 7]) == 1


def test_random_witness():
    rng = random.Random(27)
    for _ in range(500):
        a = [rng.randint(0, 15) for _ in range(rng.randint(0, 20))]
        w = m.lis(a)
        assert len(w) == brute_len(a) == m.lis_length(a)
        assert all(x < y for x, y in zip(w, w[1:])) and is_subseq(w, a)


def test_n_log_n():
    rng = random.Random(0)
    a = [rng.randint(0, 10**9) for _ in range(200_000)]
    t0 = time.perf_counter()
    m.lis_length(a)
    assert time.perf_counter() - t0 < 3
