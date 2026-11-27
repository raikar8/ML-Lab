import random

from helpers import key_of, load

m = load(key_of(__file__))


def ref(n, pre):
    done, order = set(), []
    while len(order) < n:
        avail = [c for c in range(n) if c not in done and all(b in done for a, b in pre if a == c)]
        if not avail:
            return []
        c = min(avail)
        done.add(c)
        order.append(c)
    return order


def test_examples():
    assert m.course_order(4, [(1, 0), (2, 0), (3, 1), (3, 2)]) == [0, 1, 2, 3]
    assert m.course_order(2, [(0, 1), (1, 0)]) == []
    assert m.course_order(3, []) == [0, 1, 2]
    assert m.course_order(0, []) == []
    assert not m.can_finish(1, [(0, 0)]) and m.can_finish(2, [(1, 0)])


def test_random_against_reference():
    rng = random.Random(21)
    for _ in range(400):
        n = rng.randint(1, 8)
        pre = list({(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 10))})
        assert m.course_order(n, pre) == ref(n, pre), (n, pre)


def test_large_chain():
    n = 100_000
    pre = [(i + 1, i) for i in range(n - 1)]
    assert m.course_order(n, pre)[-1] == n - 1
