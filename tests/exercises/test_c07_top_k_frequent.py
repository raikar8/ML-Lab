import ast
import inspect
import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(nums, k):
    order = list(dict.fromkeys(nums))
    return sorted(order, key=lambda x: (-nums.count(x), order.index(x)))[:k]


def test_examples():
    assert m.top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
    assert m.top_k_frequent([4, 5, 5, 4, 6], 2) == [4, 5]
    assert m.top_k_frequent([7], 3) == [7]
    assert m.top_k_frequent([], 2) == []


def test_random_against_brute_force():
    rng = random.Random(7)
    for _ in range(500):
        nums = [rng.randint(0, 8) for _ in range(rng.randint(0, 30))]
        k = rng.randint(1, 6)
        assert m.top_k_frequent(nums, k) == brute(nums, k), (nums, k)


def test_no_most_common():
    tree = ast.parse(inspect.getsource(m))
    assert not any(isinstance(n, ast.Attribute) and n.attr == "most_common" for n in ast.walk(tree))
