import itertools
import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute_once(c, t):
    return sorted({tuple(sorted(x)) for r in range(len(c) + 1) for x in itertools.combinations(c, r) if sum(x) == t})


def brute_reuse(c, t):
    out = set()

    def go(i, left, cur):
        if left == 0:
            out.add(tuple(cur)); return
        if i == len(c) or left < 0:
            return
        go(i, left - c[i], cur + [c[i]])
        go(i + 1, left, cur)

    go(0, t, [])
    return sorted(tuple(sorted(x)) for x in out)


def test_examples():
    assert m.combination_sum([2, 3, 6, 7], 7) == [[2, 2, 3], [7]]
    assert m.combination_sum_once([10, 1, 2, 7, 6, 1, 5], 8) == [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]
    assert m.combination_sum([2], 1) == [] and m.combination_sum([3], 0) == [[]]


def test_random():
    rng = random.Random(48)
    for _ in range(200):
        c = rng.sample(range(1, 10), rng.randint(1, 4))
        t = rng.randint(1, 15)
        assert [tuple(x) for x in m.combination_sum(c, t)] == brute_reuse(sorted(c), t)
        d = [rng.randint(1, 6) for _ in range(rng.randint(0, 8))]
        assert [tuple(x) for x in m.combination_sum_once(d, t)] == brute_once(d, t)
