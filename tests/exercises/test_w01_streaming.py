import random
import statistics

from helpers import key_of, load

m = load(key_of(__file__))


def test_welford_stable():
    rng = random.Random(0)
    xs = [1e9 + rng.gauss(0, 1) for _ in range(5000)]
    s = m.RunningStats()
    for x in xs:
        s.update(x)
    assert s.n == 5000
    assert abs(s.mean - statistics.fmean(xs)) < 1e-6
    assert abs(s.var - statistics.variance(xs)) / statistics.variance(xs) < 1e-6
    t = m.RunningStats(); t.update(3.0)
    assert t.var == 0.0


def test_reservoir_uniform():
    counts = [0] * 10
    rng = random.Random(1)
    for _ in range(20000):
        for x in m.reservoir_sample(iter(range(10)), 3, rng):
            counts[x] += 1
    assert all(abs(c / 20000 - 0.3) < 0.02 for c in counts)
    assert sorted(m.reservoir_sample(range(2), 5, rng)) == [0, 1]


def test_top_k():
    assert m.top_k_frequent("abracadabra", 2) == ["a", "b"]
    assert m.top_k_frequent([3, 1, 2, 2, 1, 3], 2) == [1, 2]
