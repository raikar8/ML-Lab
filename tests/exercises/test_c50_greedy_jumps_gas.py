import random
from collections import deque

from helpers import key_of, load

m = load(key_of(__file__))


def bfs_jumps(a):
    n = len(a)
    dist = [-1] * n
    dist[0] = 0
    q = deque([0])
    while q:
        i = q.popleft()
        for j in range(i + 1, min(n, i + a[i] + 1)):
            if dist[j] < 0:
                dist[j] = dist[i] + 1; q.append(j)
    return dist[-1]


def brute_gas(g, c):
    n = len(g)
    for s in range(n):
        tank, ok = 0, True
        for k in range(n):
            i = (s + k) % n
            tank += g[i] - c[i]
            if tank < 0:
                ok = False; break
        if ok:
            return s
    return -1


def test_examples():
    assert m.can_reach_end([2, 3, 1, 1, 4]) and not m.can_reach_end([3, 2, 1, 0, 4])
    assert m.min_jumps([2, 3, 1, 1, 4]) == 2 and m.min_jumps([0]) == 0 and m.min_jumps([0, 1]) == -1
    assert m.gas_start([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3 and m.gas_start([2, 3, 4], [3, 4, 3]) == -1


def test_random():
    rng = random.Random(55)
    for _ in range(600):
        a = [rng.randint(0, 3) for _ in range(rng.randint(1, 12))]
        d = bfs_jumps(a)
        assert m.can_reach_end(a) == (d >= 0) and m.min_jumps(a) == d
        n = rng.randint(1, 8)
        g = [rng.randint(0, 5) for _ in range(n)]
        c = [rng.randint(0, 5) for _ in range(n)]
        assert m.gas_start(g, c) == brute_gas(g, c)
