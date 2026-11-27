import random
import statistics

from helpers import key_of, load

m = load(key_of(__file__))


class Spy(list):
    calls = 0

    def __getitem__(self, i):
        if isinstance(i, slice):
            raise AssertionError("no slicing")
        Spy.calls += 1
        return list.__getitem__(self, i)


def test_examples():
    assert m.median_two([1, 3], [2]) == 2.0 and m.median_two([1, 2], [3, 4]) == 2.5
    assert m.median_two([], [5]) == 5.0 and m.median_two([1, 1], [1, 1]) == 1.0


def test_random():
    rng = random.Random(59)
    for _ in range(2000):
        a = sorted(rng.randint(-20, 20) for _ in range(rng.randint(0, 9)))
        b = sorted(rng.randint(-20, 20) for _ in range(rng.randint(0 if a else 1, 9)))
        assert m.median_two(a, b) == statistics.median(a + b), (a, b)


def test_logarithmic():
    a = Spy(range(0, 2_000_000, 2))
    b = Spy(range(1, 2_000_001, 2))
    Spy.calls = 0
    assert m.median_two(a, b) == 999_999.5
    assert Spy.calls < 200
