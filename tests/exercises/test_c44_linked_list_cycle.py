import ast
import inspect
import random

from helpers import key_of, load

m = load(key_of(__file__))


def nth(head, i):
    for _ in range(i):
        head = head.next
    return head


def test_cycles():
    rng = random.Random(46)
    for _ in range(300):
        n = rng.randint(0, 30)
        pos = rng.randint(-1, n - 1) if n else -1
        head = m.make_cycle(m.from_list(list(range(n))), pos)
        assert m.has_cycle(head) == (pos >= 0)
        start = m.cycle_start(head)
        assert (start is None) if pos < 0 else (start is nth(head, pos))


def test_middle():
    assert m.middle(None) is None
    for n in range(1, 12):
        head = m.from_list(list(range(n)))
        assert m.middle(head).val == n // 2


def test_find_duplicate():
    rng = random.Random(47)
    for _ in range(300):
        n = rng.randint(1, 30)
        d = rng.randint(1, n)
        nums = list(range(1, n + 1)) + [d]            # n + 1 values in [1, n], d twice
        others = [i for i, x in enumerate(nums) if x != d]
        for i in rng.sample(others, rng.randint(0, len(others))):
            nums[i] = d                               # d may repeat many times; others vanish
        rng.shuffle(nums)
        copy = list(nums)
        assert m.find_duplicate(nums) == d and nums == copy


def test_no_visited_set():
    tree = ast.parse(inspect.getsource(m))
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "set" not in calls and not any(isinstance(n, (ast.Set, ast.SetComp, ast.Dict, ast.DictComp)) for n in ast.walk(tree))
