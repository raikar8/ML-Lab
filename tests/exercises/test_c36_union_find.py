import random
import time

from helpers import key_of, load

m = load(key_of(__file__))


def test_dsu_random_against_bfs():
    rng = random.Random(36)
    for _ in range(100):
        n = rng.randint(1, 30)
        d = m.DSU(n)
        adj = {i: set() for i in range(n)}
        for _ in range(rng.randint(0, 40)):
            a, b = rng.randrange(n), rng.randrange(n)
            comp_a = reach(adj, a)
            assert d.union(a, b) == (b not in comp_a)
            adj[a].add(b); adj[b].add(a)
        comps = []
        seen = set()
        for i in range(n):
            if i not in seen:
                c = reach(adj, i); seen |= c; comps.append(c)
        assert d.components == len(comps)
        for c in comps:
            for x in c:
                assert d.size(x) == len(c) and d.connected(x, min(c))


def reach(adj, s):
    seen, st = {s}, [s]
    while st:
        for v in adj[st.pop()]:
            if v not in seen:
                seen.add(v); st.append(v)
    return seen


def test_chain_is_fast():
    n = 200_000
    d = m.DSU(n)
    t0 = time.perf_counter()
    for i in range(n - 1):
        d.union(i, i + 1)
    for i in range(0, n, 7):
        d.find(i)
    assert d.components == 1 and d.size(0) == n and time.perf_counter() - t0 < 3


def test_redundant_edge():
    assert m.redundant_edge(3, [(0, 1), (0, 2), (1, 2)]) == (1, 2)
    assert m.redundant_edge(5, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 4)]) == (0, 3)


def test_merge_accounts():
    acc = [["John", "j@x", "john@y"], ["John", "j2@z"], ["John", "john@y", "jj@w"], ["Mary", "m@q"], ["Ann"]]
    assert m.merge_accounts(acc) == [["Ann"], ["John", "j2@z"], ["John", "j@x", "jj@w", "john@y"], ["Mary", "m@q"]]
