import random

from helpers import key_of, load

m = load(key_of(__file__))


def covered(intervals):
    pts = set()
    for s, e in intervals:
        pts.update(x / 2 for x in range(2 * s, 2 * e + 1))
    return pts


def check_merged(out):
    for a, b in zip(out, out[1:]):
        assert a[1] < b[0]


def test_examples():
    assert m.merge([[1, 3], [8, 10], [2, 6], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert m.merge([[1, 3], [3, 5]]) == [[1, 5]]
    assert m.merge([]) == []
    assert m.merge([[1, 10], [2, 3]]) == [[1, 10]]
    assert m.insert([[1, 2], [3, 5], [6, 7], [8, 10]], [4, 8]) == [[1, 2], [3, 10]]
    assert m.insert([], [2, 3]) == [[2, 3]]
    assert m.insert([[5, 6]], [1, 2]) == [[1, 2], [5, 6]]


def test_random():
    rng = random.Random(10)
    for _ in range(400):
        iv = []
        for _ in range(rng.randint(0, 8)):
            s = rng.randint(0, 20)
            iv.append([s, s + rng.randint(0, 4)])
        out = m.merge([list(x) for x in iv])
        check_merged(out)
        assert covered(out) == covered(iv)
        s = rng.randint(0, 20)
        new = [s, s + rng.randint(0, 6)]
        ins = m.insert(out, new)
        check_merged(ins)
        assert covered(ins) == covered(iv + [new])
