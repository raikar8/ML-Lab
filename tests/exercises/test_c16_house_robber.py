import itertools
import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(v, circle):
    n, best = len(v), 0
    for mask in itertools.product([0, 1], repeat=n):
        if any(mask[i] and mask[i + 1] for i in range(n - 1)):
            continue
        if circle and n > 1 and mask[0] and mask[-1]:
            continue
        best = max(best, sum(x for x, t in zip(v, mask) if t))
    return best


def test_examples():
    assert m.rob_line([2, 7, 9, 3, 1]) == 12
    assert m.rob_circle([2, 3, 2]) == 3 and m.rob_circle([1, 2, 3, 1]) == 4
    assert m.rob_line([]) == 0 and m.rob_circle([]) == 0 and m.rob_circle([5]) == 5


def test_random_against_brute_force():
    rng = random.Random(16)
    for _ in range(400):
        v = [rng.randint(0, 9) for _ in range(rng.randint(0, 11))]
        assert m.rob_line(v) == brute(v, False) and m.rob_circle(v) == brute(v, True), v
