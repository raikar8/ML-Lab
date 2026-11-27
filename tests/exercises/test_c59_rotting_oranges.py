import copy
import random

from helpers import key_of, load

m = load(key_of(__file__))


def sim(grid):
    g = copy.deepcopy(grid)
    t = 0
    while True:
        R, C = len(g), len(g[0]) if g else 0
        new = [(r, c) for r in range(R) for c in range(C) if g[r][c] == 1 and any(
            0 <= r + dr < R and 0 <= c + dc < C and g[r + dr][c + dc] == 2 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
        if not new:
            return -1 if any(1 in row for row in g) else t
        for r, c in new:
            g[r][c] = 2
        t += 1


def test_examples():
    assert m.minutes_to_rot([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
    assert m.minutes_to_rot([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1
    assert m.minutes_to_rot([[0, 2]]) == 0 and m.minutes_to_rot([[1]]) == -1
    assert m.nearest_zero([[0, 0, 0], [0, 1, 0], [1, 1, 1]]) == [[0, 0, 0], [0, 1, 0], [1, 2, 1]]


def test_random():
    rng = random.Random(61)
    for _ in range(400):
        R, C = rng.randint(1, 6), rng.randint(1, 6)
        g = [[rng.choice([0, 1, 1, 2]) for _ in range(C)] for _ in range(R)]
        before = copy.deepcopy(g)
        assert m.minutes_to_rot(g) == sim(g) and g == before
        mat = [[rng.choice([0, 1, 1]) for _ in range(C)] for _ in range(R)]
        mat[rng.randrange(R)][rng.randrange(C)] = 0
        zeros = [(r, c) for r in range(R) for c in range(C) if mat[r][c] == 0]
        exp = [[min(abs(r - a) + abs(c - b) for a, b in zeros) for c in range(C)] for r in range(R)]
        assert m.nearest_zero(mat) == exp
