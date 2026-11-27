import collections
import math
import random

import pytest

from helpers import key_of, load

m = load(key_of(__file__))


def chi2_ok(counts, probs, n):
    stat = sum((counts.get(i, 0) - n * p) ** 2 / (n * p) for i, p in enumerate(probs) if p > 0)
    return stat < 3 * len(probs) + 30   # very loose; catches real bias


def test_weighted_pick_distribution():
    w = [1, 0, 3, 6]
    p = m.WeightedPicker(w)
    rng = random.Random(50)
    n = 60_000
    c = collections.Counter(p.pick(rng) for _ in range(n))
    assert c[1] == 0 and chi2_ok(c, [x / 10 for x in w], n)
    with pytest.raises(ValueError):
        m.WeightedPicker([0, 0])


def test_one_random_call_per_pick():
    class Counting(random.Random):
        calls = 0

        def random(self):
            Counting.calls += 1
            return super().random()

    p = m.WeightedPicker([2, 2, 2])
    r = Counting(1)
    for _ in range(100):
        p.pick(r)
    assert Counting.calls == 100


def test_shuffle_uniform():
    rng = random.Random(51)
    c = collections.Counter()
    n = 60_000
    for _ in range(n):
        a = [0, 1, 2, 3]
        m.shuffle(a, rng)
        c[tuple(a)] += 1
    assert len(c) == 24 and chi2_ok({i: v for i, v in enumerate(c.values())}, [1 / 24] * 24, n)


def test_weighted_sample():
    rng = random.Random(52)
    w = [5, 1, 0, 4]
    n = 40_000
    first = collections.Counter()
    for _ in range(n):
        s = m.weighted_sample(w, 2, rng)
        assert len(set(s)) == 2 and 2 not in s
        first[tuple(sorted(s))] += 1
    # P({0,1}) = 5/10*1/5 + 1/10*5/9 ; etc.
    def pair(a, b):
        return w[a] / 10 * w[b] / (10 - w[a]) + w[b] / 10 * w[a] / (10 - w[b])
    for a, b in [(0, 1), (0, 3), (1, 3)]:
        assert abs(first[(a, b)] / n - pair(a, b)) < 0.015
    with pytest.raises(ValueError):
        m.weighted_sample([1, 0, 0], 2, rng)
