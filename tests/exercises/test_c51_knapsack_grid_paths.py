import functools
import itertools
import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute_knap(w, v, cap):
    best = 0
    for mask in itertools.product([0, 1], repeat=len(w)):
        if sum(a * b for a, b in zip(w, mask)) <= cap:
            best = max(best, sum(a * b for a, b in zip(v, mask)))
    return best


def test_knapsack():
    assert m.knapsack([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9
    assert m.knapsack([], [], 5) == 0 and m.knapsack([6], [10], 5) == 0
    rng = random.Random(56)
    for _ in range(300):
        n = rng.randint(0, 8)
        w = [rng.randint(1, 8) for _ in range(n)]
        v = [rng.randint(0, 9) for _ in range(n)]
        cap = rng.randint(0, 20)
        exp = brute_knap(w, v, cap)
        assert m.knapsack(w, v, cap) == exp
        best, items = m.knapsack_items(w, v, cap)
        assert best == exp and sum(w[i] for i in items) <= cap and sum(v[i] for i in items) == exp
        assert items == sorted(set(items))


def test_grid_paths():
    assert m.grid_paths(["...", ".#.", "..."]) == 2 and m.grid_paths(["#"]) == 0 and m.grid_paths(["."]) == 1
    rng = random.Random(57)
    for _ in range(300):
        R, C = rng.randint(1, 6), rng.randint(1, 6)
        g = ["".join(rng.choice("...#") for _ in range(C)) for _ in range(R)]

        @functools.lru_cache(None)
        def go(r, c):
            if r >= R or c >= C or g[r][c] == "#":
                return 0
            if (r, c) == (R - 1, C - 1):
                return 1
            return go(r + 1, c) + go(r, c + 1)

        assert m.grid_paths(g) == go(0, 0)


def test_min_path_sum():
    assert m.min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 7 and m.min_path_sum([[5]]) == 5
    rng = random.Random(58)
    for _ in range(200):
        R, C = rng.randint(1, 5), rng.randint(1, 5)
        a = [[rng.randint(0, 9) for _ in range(C)] for _ in range(R)]

        @functools.lru_cache(None)
        def go(r, c):
            if r == R - 1 and c == C - 1:
                return a[r][c]
            opts = []
            if r + 1 < R:
                opts.append(go(r + 1, c))
            if c + 1 < C:
                opts.append(go(r, c + 1))
            return a[r][c] + min(opts)

        assert m.min_path_sum(a) == go(0, 0)
