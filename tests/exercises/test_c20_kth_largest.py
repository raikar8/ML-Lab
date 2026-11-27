import ast
import inspect
import random
import time

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    assert m.kth_largest_heap([3, 2, 1, 5, 6, 4], 2) == 5
    assert m.kth_largest_select([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
    assert m.kth_largest_select([7], 1) == 7 and m.kth_largest_heap([7], 1) == 7


def test_random_against_sort_and_no_mutation():
    rng = random.Random(20)
    for _ in range(500):
        a = [rng.randint(-10, 10) for _ in range(rng.randint(1, 30))]
        k = rng.randint(1, len(a))
        copy = list(a)
        exp = sorted(a, reverse=True)[k - 1]
        assert m.kth_largest_heap(a, k) == exp and m.kth_largest_select(a, k, seed=k) == exp
        assert a == copy


def test_duplicates_and_sorted_input_fast():
    t0 = time.perf_counter()
    assert m.kth_largest_select([5] * 200_000, 777) == 5
    assert m.kth_largest_select(list(range(200_000)), 1) == 199_999
    assert time.perf_counter() - t0 < 3


def test_no_shortcuts():
    tree = ast.parse(inspect.getsource(m))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} |         {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "nlargest" not in names and "sorted" not in names
