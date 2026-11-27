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


def all_nodes(r):
    out, st = [], [r]
    while st:
        n = st.pop()
        if n:
            out.append(n); st += [n.left, n.right]
    return out


def path(root, target):
    st = [(root, [root])]
    while st:
        n, p = st.pop()
        if n is target:
            return p
        for c in (n.left, n.right):
            if c:
                st.append((c, p + [c]))


def ref_lca(root, p, q):
    a, b = path(root, p), path(root, q)
    i = 0
    while i < min(len(a), len(b)) and a[i] is b[i]:
        i += 1
    return a[i - 1]


def test_examples():
    root = m.from_level([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    five, one, four = root.left, root.right, root.left.right.right
    assert m.lca(root, five, one) is root and m.lca(root, five, four) is five
    bst = m.from_level([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5])
    assert m.lca_bst(bst, bst.left, bst.right) is bst
    assert m.lca_bst(bst, bst.left, bst.left.right) is bst.left


def test_random():
    rng = random.Random(23)
    for _ in range(200):
        root = rand_tree(rng, rng.randint(1, 30))
        ns = all_nodes(root)
        for _ in range(5):
            p, q = rng.choice(ns), rng.choice(ns)
            assert m.lca(root, p, q) is ref_lca(root, p, q)


def test_deep_chain():
    root = m.TreeNode(0)
    cur = root
    nodes = [root]
    for i in range(1, 3000):
        cur.left = m.TreeNode(i); cur = cur.left; nodes.append(cur)
    assert m.lca(root, nodes[2999], nodes[1500]) is nodes[1500]
