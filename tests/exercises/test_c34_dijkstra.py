import random

from helpers import key_of, load

m = load(key_of(__file__))

INF = float("inf")


def bellman(n, edges, s):
    d = [INF] * n
    d[s] = 0
    for _ in range(n):
        for u, v, w in edges:
            if d[u] + w < d[v]:
                d[v] = d[u] + w
    return d


def test_examples():
    e = [(2, 1, 1), (2, 3, 1), (3, 4, 1)]
    assert m.network_delay(5, e, 2) == -1           # node 0 unreachable
    assert m.network_delay(4, [(0, 1, 1), (1, 2, 2), (0, 2, 5), (2, 3, 1)], 0) == 4
    d, p = m.shortest_path(4, [(0, 1, 1), (1, 2, 2), (0, 2, 5), (2, 3, 1)], 0, 3)
    assert d == 4 and p == [0, 1, 2, 3]
    assert m.shortest_path(2, [], 0, 1) == (INF, []) and m.shortest_path(1, [], 0, 0) == (0, [0])


def test_random_against_bellman_ford():
    rng = random.Random(34)
    for _ in range(300):
        n = rng.randint(1, 9)
        edges = [(rng.randrange(n), rng.randrange(n), rng.randint(0, 9)) for _ in range(rng.randint(0, 20))]
        s, t = rng.randrange(n), rng.randrange(n)
        ref = bellman(n, edges, s)
        assert m.dijkstra(n, edges, s) == ref
        d, path = m.shortest_path(n, edges, s, t)
        assert d == ref[t]
        if d < INF:
            assert path[0] == s and path[-1] == t
            w = {}
            for u, v, c in edges:
                w[(u, v)] = min(w.get((u, v), INF), c)
            assert sum(w[(a, b)] for a, b in zip(path, path[1:])) == d


def test_large_sparse():
    n = 50_000
    edges = [(i, i + 1, 1) for i in range(n - 1)] + [(0, n - 1, 10**9)]
    assert m.network_delay(n, edges, 0) == n - 1
