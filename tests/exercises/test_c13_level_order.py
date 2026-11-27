import random

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


def ref_levels(root):
    out = []

    def dfs(n, d):
        if n is None:
            return
        if d == len(out):
            out.append([])
        out[d].append(n.val)
        dfs(n.left, d + 1)
        dfs(n.right, d + 1)

    dfs(root, 0)
    return out


def test_examples():
    root = m.from_level([3, 9, 20, None, None, 15, 7])
    assert m.level_order(root) == [[3], [9, 20], [15, 7]]
    assert m.zigzag(root) == [[3], [20, 9], [15, 7]]
    assert m.right_view(root) == [3, 20, 7]
    assert m.level_order(None) == [] and m.zigzag(None) == [] and m.right_view(None) == []
    left_only = m.from_level([1, 2, None, 3])
    assert m.right_view(left_only) == [1, 2, 3]


def test_random_trees():
    rng = random.Random(13)
    for _ in range(300):
        root = rand_tree(rng, rng.randint(0, 25))
        lv = ref_levels(root)
        assert m.level_order(root) == lv
        assert m.zigzag(root) == [x if i % 2 == 0 else x[::-1] for i, x in enumerate(lv)]
        assert m.right_view(root) == [x[-1] for x in lv]


def test_deep_tree_no_recursion_limit():
    root = m.TreeNode(0)
    cur = root
    for i in range(1, 5000):
        cur.right = m.TreeNode(i)
        cur = cur.right
    assert len(m.level_order(root)) == 5000
