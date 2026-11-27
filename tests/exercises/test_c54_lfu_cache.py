import ast
import inspect
import random
import time

from helpers import key_of, load

m = load(key_of(__file__))


class RefLFU:
    def __init__(self, cap):
        self.cap, self.d, self.t = cap, {}, 0   # key -> [value, freq, last_used]

    def get(self, k):
        if k not in self.d:
            return -1
        self.t += 1; self.d[k][1] += 1; self.d[k][2] = self.t
        return self.d[k][0]

    def put(self, k, v):
        if self.cap == 0:
            return
        self.t += 1
        if k in self.d:
            self.d[k][0] = v; self.d[k][1] += 1; self.d[k][2] = self.t; return
        if len(self.d) >= self.cap:
            victim = min(self.d, key=lambda x: (self.d[x][1], self.d[x][2]))
            del self.d[victim]
        self.d[k] = [v, 1, self.t]


def test_example():
    c = m.LFUCache(2)
    c.put(1, 1); c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)                       # evicts 2 (freq 1) — 1 has freq 2
    assert c.get(2) == -1 and c.get(3) == 3
    c.put(4, 4)                       # 1 and 3 both freq 2; 1 is least recently used
    assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4 and len(c) == 2
    z = m.LFUCache(0); z.put(1, 1)
    assert z.get(1) == -1 and len(z) == 0


def test_random_against_reference():
    rng = random.Random(60)
    for cap in (1, 2, 3, 5):
        c, r = m.LFUCache(cap), RefLFU(cap)
        for _ in range(3000):
            k = rng.randint(0, 7)
            if rng.random() < 0.5:
                assert c.get(k) == r.get(k)
            else:
                v = rng.randint(0, 99); c.put(k, v); r.put(k, v)
            assert len(c) == len(r.d)


def test_constant_time_and_no_heap():
    tree = ast.parse(inspect.getsource(m))
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "heapq" not in mods
    c = m.LFUCache(50_000)
    t0 = time.perf_counter()
    for i in range(200_000):
        c.put(i, i)
        c.get(i - 3)
    assert time.perf_counter() - t0 < 3
