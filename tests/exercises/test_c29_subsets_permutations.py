import ast
import inspect
import itertools
import random

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    assert m.subsets([1, 2, 2]) == [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
    assert m.permutations([1, 1, 2]) == [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
    assert m.subsets([]) == [[]] and m.permutations([]) == [[]]


def test_random_against_itertools():
    rng = random.Random(29)
    for _ in range(150):
        a = [rng.randint(0, 3) for _ in range(rng.randint(0, 6))]
        subs = sorted({tuple(sorted(c)) for r in range(len(a) + 1) for c in itertools.combinations(a, r)})
        perms = sorted(set(itertools.permutations(a)))
        assert m.subsets(a) == [list(s) for s in subs]
        assert m.permutations(a) == [list(p) for p in perms]


def test_no_itertools():
    tree = ast.parse(inspect.getsource(m))
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "itertools" not in mods
