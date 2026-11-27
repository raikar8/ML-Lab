import random

from helpers import key_of, load

m = load(key_of(__file__))


def nodes(head):
    out = []
    while head:
        out.append(head)
        head = head.next
    return out


def test_reverse():
    assert m.to_list(m.reverse(m.from_list([1, 2, 3]))) == [3, 2, 1]
    assert m.reverse(None) is None
    assert m.to_list(m.reverse(m.from_list([7]))) == [7]


def test_reverse_relinks_same_nodes():
    head = m.from_list(list(range(6)))
    before = {id(n) for n in nodes(head)}
    out = m.reverse(head)
    assert {id(n) for n in nodes(out)} == before and m.to_list(out) == [5, 4, 3, 2, 1, 0]


def test_k_group():
    assert m.to_list(m.reverse_k_group(m.from_list([1, 2, 3, 4, 5]), 2)) == [2, 1, 4, 3, 5]
    assert m.to_list(m.reverse_k_group(m.from_list([1, 2, 3, 4, 5]), 3)) == [3, 2, 1, 4, 5]
    assert m.to_list(m.reverse_k_group(m.from_list([1, 2]), 5)) == [1, 2]
    assert m.to_list(m.reverse_k_group(m.from_list([1, 2, 3]), 1)) == [1, 2, 3]
    rng = random.Random(6)
    for _ in range(200):
        vals = [rng.randint(0, 9) for _ in range(rng.randint(0, 15))]
        k = rng.randint(1, 5)
        exp = []
        for i in range(0, len(vals), k):
            blk = vals[i:i + k]
            exp += blk[::-1] if len(blk) == k else blk
        head = m.from_list(vals)
        ids = sorted(id(n) for n in nodes(head))
        out = m.reverse_k_group(head, k)
        assert m.to_list(out) == exp and sorted(id(n) for n in nodes(out)) == ids


def test_long_list_iterative():
    head = m.from_list(list(range(100_000)))
    assert m.to_list(m.reverse(head))[:3] == [99_999, 99_998, 99_997]
