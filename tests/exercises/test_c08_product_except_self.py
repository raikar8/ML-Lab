import inspect
import math
import random

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    assert m.product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert m.product_except_self([0, 4, 0]) == [0, 0, 0]
    assert m.product_except_self([2, 0, 3]) == [0, 6, 0]
    assert m.product_except_self([5]) == [1]
    assert m.product_except_self([]) == []


def test_random():
    rng = random.Random(8)
    for _ in range(500):
        nums = [rng.randint(-3, 3) for _ in range(rng.randint(1, 10))]
        exp = [math.prod(nums[:i] + nums[i + 1:]) for i in range(len(nums))]
        assert m.product_except_self(nums) == exp


def test_no_division():
    src = inspect.getsource(m.product_except_self)
    assert "/" not in src.split('"""')[-1] and "//" not in src
