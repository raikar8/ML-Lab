import random

from helpers import key_of, load

m = load(key_of(__file__))


class Spy(list):
    calls = 0

    def __getitem__(self, i):
        Spy.calls += 1
        return list.__getitem__(self, i)


def test_examples():
    a = [4, 5, 6, 7, 0, 1, 2]
    assert m.find_min_index(a) == 4 and m.search(a, 0) == 4 and m.search(a, 3) == -1
    assert m.find_min_index([1]) == 0 and m.search([1], 1) == 0 and m.search([], 1) == -1
    assert m.find_min_index([1, 2, 3]) == 0 and m.search([3, 1], 1) == 1


def test_all_rotations():
    rng = random.Random(18)
    for _ in range(200):
        base = sorted(rng.sample(range(-50, 50), rng.randint(1, 15)))
        r = rng.randrange(len(base))
        a = base[r:] + base[:r]
        assert m.find_min_index(a) == (len(base) - r) % len(base)
        for t in range(-52, 52):
            assert m.search(a, t) == (a.index(t) if t in a else -1)


def test_logarithmic():
    n = 1 << 20
    a = Spy(list(range(n // 3, n)) + list(range(n // 3)))
    Spy.calls = 0
    assert m.search(a, 12345) == a.index(12345)
    assert Spy.calls < 200
