import random

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    assert m.days_until_warmer([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert m.next_greater([2, 1, 2, 4, 3]) == [4, 2, 4, -1, -1]
    assert m.days_until_warmer([]) == [] and m.days_until_warmer([5, 5]) == [0, 0]


def test_random_against_brute_force():
    rng = random.Random(15)
    for _ in range(500):
        a = [rng.randint(0, 9) for _ in range(rng.randint(0, 20))]
        exp_d = [next((j - i for j in range(i + 1, len(a)) if a[j] > a[i]), 0) for i in range(len(a))]
        exp_g = [next((a[j] for j in range(i + 1, len(a)) if a[j] > a[i]), -1) for i in range(len(a))]
        assert m.days_until_warmer(a) == exp_d and m.next_greater(a) == exp_g


def test_linear_on_decreasing_then_spike():
    a = list(range(200_000, 0, -1)) + [10**9]
    out = m.days_until_warmer(a)
    assert out[0] == 200_000 and out[-1] == 0
