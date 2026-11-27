import random

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    assert m.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6 and m.trap([4, 2, 0, 3, 2, 5]) == 9
    assert m.trap([]) == 0 and m.trap([3]) == 0
    assert m.max_container([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49 and m.max_container([5]) == 0


def test_random_against_brute_force():
    rng = random.Random(44)
    for _ in range(600):
        h = [rng.randint(0, 6) for _ in range(rng.randint(0, 15))]
        exp = sum(max(0, min(max(h[:i + 1]), max(h[i:])) - h[i]) for i in range(len(h)))
        assert m.trap(h) == exp
        exp_c = max((min(h[i], h[j]) * (j - i) for i in range(len(h)) for j in range(i + 1, len(h))), default=0)
        assert m.max_container(h) == exp_c
