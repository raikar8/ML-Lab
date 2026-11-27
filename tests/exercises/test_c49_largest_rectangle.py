import random
import time

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    assert m.largest_rectangle([2, 1, 5, 6, 2, 3]) == 10 and m.largest_rectangle([]) == 0
    assert m.largest_rectangle([2, 4]) == 4 and m.largest_rectangle([3, 3, 3]) == 9
    assert m.maximal_rectangle(["10100", "10111", "11111", "10010"]) == 6 and m.maximal_rectangle([]) == 0


def test_random_against_brute_force():
    rng = random.Random(54)
    for _ in range(500):
        h = [rng.randint(0, 6) for _ in range(rng.randint(0, 12))]
        exp = max((min(h[i:j]) * (j - i) for i in range(len(h)) for j in range(i + 1, len(h) + 1)), default=0)
        assert m.largest_rectangle(h) == exp
    for _ in range(200):
        R, C = rng.randint(1, 5), rng.randint(1, 5)
        g = ["".join(rng.choice("01") for _ in range(C)) for _ in range(R)]
        exp = 0
        for r1 in range(R):
            for r2 in range(r1, R):
                for c1 in range(C):
                    for c2 in range(c1, C):
                        if all(g[r][c] == "1" for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)):
                            exp = max(exp, (r2 - r1 + 1) * (c2 - c1 + 1))
        assert m.maximal_rectangle(g) == exp


def test_linear():
    h = list(range(200_000))
    t0 = time.perf_counter()
    assert m.largest_rectangle(h) == max(v * (200_000 - v) for v in h)
    assert time.perf_counter() - t0 < 2
