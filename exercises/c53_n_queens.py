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
    raise NotImplementedError  # TODO: implement


def solve_queens(n: int) -> list[list[str]]:
    raise NotImplementedError  # TODO: implement
