import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(points, q, k):
    order = sorted(range(len(points)), key=lambda i: (sum((a - b) ** 2 for a, b in zip(points[i], q)), i))
    return [points[i] for i in order[:k]]


def test_examples():
    assert m.k_closest([(1, 3), (-2, 2), (5, 8), (0, 1)], (0, 0), 2) == [(0, 1), (-2, 2)]
    assert m.k_closest([], (0, 0), 3) == [] and m.k_closest([(1, 1)], (0, 0), 0) == []


def test_random_with_ties():
    rng = random.Random(31)
    for _ in range(400):
        dim = rng.randint(1, 3)
        pts = [tuple(rng.randint(-3, 3) for _ in range(dim)) for _ in range(rng.randint(0, 25))]
        q = tuple(rng.randint(-2, 2) for _ in range(dim))
        k = rng.randint(1, 30)
        assert m.k_closest(pts, q, k) == brute(pts, q, k)
