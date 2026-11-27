import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(nums, target):
    for j in range(len(nums)):
        for i in range(j):
            if nums[i] + nums[j] == target:
                return (i, j)
    return None


def test_examples():
    assert m.two_sum([3, 1, 4, 1, 5], 2) == (1, 3)
    assert m.two_sum([2, 7, 11, 15], 9) == (0, 1)
    assert m.two_sum([1, 2], 7) is None
    assert m.two_sum([], 0) is None
    assert m.two_sum([5], 10) is None          # an element can't pair with itself
    assert m.two_sum([-3, 4, 3, 90], 0) == (0, 2)


def test_random_against_brute_force():
    rng = random.Random(0)
    for _ in range(400):
        nums = [rng.randint(-20, 20) for _ in range(rng.randint(0, 25))]
        t = rng.randint(-30, 30)
        assert m.two_sum(nums, t) == brute(nums, t), (nums, t)


def test_linear_time():
    nums = list(range(200_000))
    assert m.two_sum(nums, 399_997) == (199_998, 199_999)
