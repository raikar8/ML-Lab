"""
title: N-Queens — count and list solutions with bitmask backtracking
pattern: Backtracking
level: hard
time: 35
frequency: medium
week: 27
day: 2
tags: backtracking, bitmask, pruning
complexity: roughly O(n!) worst case, far less with pruning; O(n) state with bitmasks
hint: Place one queen per row. Track used columns and both diagonals (r + c and r − c) to test a square in O(1).
hint: Bitmasks: free = ~(cols | d1 | d2) & full; take the lowest bit with free & −free; shift d1/d2 for the next row.
follow-up: Exploit symmetry (mirror the first row) to halve the work.
follow-up: Sudoku solver — same constraint-propagation + backtracking ideas.

* `count_queens(n)` — number of ways to place `n` non-attacking queens on an `n×n` board; must do `n = 12`
  (14 200 solutions) in a few seconds.
* `solve_queens(n)` — all solutions, each as a list of `n` strings with `'Q'` and `'.'`, sorted lexicographically
  (by the list of strings).
"""


def count_queens(n: int) -> int:
    full = (1 << n) - 1

    def go(cols, d1, d2):
        if cols == full:
            return 1
        total = 0
        free = ~(cols | d1 | d2) & full
        while free:
            bit = free & -free
            free ^= bit
            total += go(cols | bit, ((d1 | bit) << 1) & full, (d2 | bit) >> 1)
        return total

    return go(0, 0, 0)


def solve_queens(n: int) -> list[list[str]]:
    out = []
    cols, d1, d2, pos = set(), set(), set(), []

    def go(r):
        if r == n:
            out.append(["." * c + "Q" + "." * (n - c - 1) for c in pos])
            return
        for c in range(n):
            if c in cols or r + c in d1 or r - c in d2:
                continue
            cols.add(c); d1.add(r + c); d2.add(r - c); pos.append(c)
            go(r + 1)
            cols.discard(c); d1.discard(r + c); d2.discard(r - c); pos.pop()

    go(0)
    return sorted(out)
