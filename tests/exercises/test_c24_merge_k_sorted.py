import ast
import inspect
import itertools
import random

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    assert m.merge_k([[1, 4, 5], [1, 3, 4], [2, 6]]) == [1, 1, 2, 3, 4, 4, 5, 6]
    assert m.merge_k([]) == [] and m.merge_k([[], []]) == [] and m.merge_k([[2], []]) == [2]


def test_random():
    rng = random.Random(24)
    for _ in range(300):
        ls = [sorted(rng.randint(0, 20) for _ in range(rng.randint(0, 8))) for _ in range(rng.randint(0, 6))]
        assert m.merge_k(ls) == sorted(itertools.chain(*ls))


def test_lazy_on_infinite_and_stable():
    evens = itertools.count(0, 2)
    odds = itertools.count(1, 2)
    assert list(itertools.islice(m.merge_k_iters([evens, odds]), 7)) == [0, 1, 2, 3, 4, 5, 6]

    class Tagged:
        def __init__(self, v, tag):
            self.v, self.tag = v, tag

        def __lt__(self, other):
            return self.v < other.v

        def __eq__(self, other):
            return self.v == other.v

    a = [Tagged(1, "a"), Tagged(2, "a")]
    b = [Tagged(1, "b"), Tagged(2, "b")]
    assert [t.tag for t in m.merge_k_iters([a, b])] == ["a", "b", "a", "b"]


def test_reads_one_ahead():
    pulled = []

    def gen(i):
        for v in range(i, 100, 3):
            pulled.append(i)
            yield v

    it = m.merge_k_iters([gen(0), gen(1), gen(2)])
    next(it)
    assert len(pulled) <= 4


def test_no_shortcuts():
    tree = ast.parse(inspect.getsource(m))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "sorted" not in names and not ({"merge"} & {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})
