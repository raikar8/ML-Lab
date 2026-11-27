import ast
import inspect
import random

from helpers import key_of, load

m = load(key_of(__file__))


def same(a, b):
    st = [(a, b)]
    while st:
        x, y = st.pop()
        if (x is None) != (y is None):
            return False
        if x is None:
            continue
        if x.val != y.val:
            return False
        st += [(x.left, y.left), (x.right, y.right)]
    return True


def rand_tree(rng, n):
    if n == 0:
        return None
    root = m.TreeNode(rng.randint(-1000, 1000))
    nodes = [root]
    for _ in range(n - 1):
        while True:
            p = rng.choice(nodes); side = rng.choice(("left", "right"))
            if getattr(p, side) is None:
                c = m.TreeNode(rng.randint(-1000, 1000)); setattr(p, side, c); nodes.append(c); break
    return root


def test_roundtrip_random():
    rng = random.Random(49)
    for _ in range(300):
        t = rand_tree(rng, rng.randint(0, 40))
        s = m.serialize(t)
        assert isinstance(s, str) and same(m.deserialize(s), t)


def test_deep_and_empty():
    assert m.deserialize(m.serialize(None)) is None
    root = m.TreeNode(0); cur = root
    for i in range(1, 10_000):
        cur.left = m.TreeNode(-i); cur = cur.left
    assert same(m.deserialize(m.serialize(root)), root)


def test_no_shortcuts():
    tree = ast.parse(inspect.getsource(m))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not names & {"eval", "repr"} and not mods & {"pickle", "json"}
