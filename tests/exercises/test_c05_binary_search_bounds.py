import ast
import bisect
import inspect
import random

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    a = [1, 2, 2, 2, 5, 7]
    assert m.lower_bound(a, 2) == 1 and m.upper_bound(a, 2) == 4
    assert m.search_range(a, 2) == (1, 3) and m.search_range(a, 3) == (-1, -1)
    assert m.count(a, 2) == 3 and m.count(a, 8) == 0
    assert m.lower_bound([], 1) == 0 and m.search_range([], 1) == (-1, -1)
    assert m.lower_bound(a, 0) == 0 and m.upper_bound(a, 99) == 6


def test_random_against_bisect():
    rng = random.Random(4)
    for _ in range(1000):
        a = sorted(rng.randint(-5, 5) for _ in range(rng.randint(0, 20)))
        x = rng.randint(-7, 7)
        assert m.lower_bound(a, x) == bisect.bisect_left(a, x)
        assert m.upper_bound(a, x) == bisect.bisect_right(a, x)
        assert m.count(a, x) == a.count(x)


def test_no_bisect_and_logarithmic():
    tree = ast.parse(inspect.getsource(m))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "bisect" not in imported
    calls = 0

    class Spy(list):
        def __getitem__(self, i):
            nonlocal calls
            calls += 1
            return list.__getitem__(self, i)

    big = Spy(range(1_000_000))
    assert m.lower_bound(big, 765_432) == 765_432
    assert calls <= 25
