import random

import pytest

from helpers import key_of, load

m = load(key_of(__file__))


def rand_tree(rng, n, lo=0, hi=50):
    if n == 0:
        return None
    vals = [rng.randint(lo, hi) for _ in range(n)]
    root = m.TreeNode(vals[0])
    nodes = [root]
    for v in vals[1:]:
        while True:
            p = rng.choice(nodes)
            side = rng.choice(("left", "right"))
            if getattr(p, side) is None:
                c = m.TreeNode(v)
                setattr(p, side, c)
                nodes.append(c)
                break
    return root


def inorder(n, out):
    if n:
        inorder(n.left, out); out.append(n.val); inorder(n.right, out)
    return out


def bst_from(vals):
    root = None
    for v in vals:
        if root is None:
            root = m.TreeNode(v); continue
        cur = root
        while True:
            side = "left" if v < cur.val else "right"
            nxt = getattr(cur, side)
            if nxt is None:
                setattr(cur, side, m.TreeNode(v)); break
            cur = nxt
    return root


def test_examples():
    assert m.is_bst(m.from_level([5, 1, 7, None, None, 6, 8]))
    assert not m.is_bst(m.from_level([5, 1, 7, None, None, 4, 8]))
    assert m.is_bst(None)
    assert not m.is_bst(m.from_level([2, 2]))                  # strict
    root = m.from_level([5, 3, 6, 2, 4, None, None, 1])
    assert [m.kth_smallest(root, k) for k in range(1, 7)] == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        m.kth_smallest(root, 7)
    with pytest.raises(ValueError):
        m.kth_smallest(root, 0)


def test_random():
    rng = random.Random(17)
    for _ in range(400):
        t = rand_tree(rng, rng.randint(0, 12), 0, 15)
        vals = inorder(t, [])
        assert m.is_bst(t) == all(a < b for a, b in zip(vals, vals[1:]))
        vals = rng.sample(range(100), rng.randint(1, 20))
        b = bst_from(vals)
        assert m.is_bst(b)
        k = rng.randint(1, len(vals))
        assert m.kth_smallest(b, k) == sorted(vals)[k - 1]
