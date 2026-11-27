import itertools
import time

from helpers import key_of, load

m = load(key_of(__file__))

KNOWN = {1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724}


def valid(board):
    n = len(board)
    pos = [row.index("Q") for row in board]
    return all(row.count("Q") == 1 and len(row) == n for row in board) and len(set(pos)) == n and         all(abs(pos[i] - pos[j]) != j - i for i in range(n) for j in range(i + 1, n))


def test_counts():
    for n, c in KNOWN.items():
        assert m.count_queens(n) == c


def test_solutions():
    for n in range(1, 8):
        sols = m.solve_queens(n)
        assert len(sols) == KNOWN[n] and sols == sorted(sols) and all(valid(s) for s in sols)
        brute = sorted(["." * c + "Q" + "." * (n - c - 1) for c in p] for p in itertools.permutations(range(n))
                       if all(abs(p[i] - p[j]) != j - i for i in range(n) for j in range(i + 1, n)))
        assert sols == brute


def test_fast_12():
    t0 = time.perf_counter()
    assert m.count_queens(12) == 14200
    assert time.perf_counter() - t0 < 8
