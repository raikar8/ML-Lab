"""
title: Matrix manipulation — rotate in place, spiral order, set zeroes
pattern: Matrix
level: medium
time: 30
frequency: high
week: 20
day: 2
tags: matrix, in-place, simulation
complexity: O(n²) / O(R·C) time; O(1) extra space for rotate and set_zeroes
hint: Rotate 90° clockwise = transpose, then reverse every row.
hint: Spiral: shrink four boundaries (top, bottom, left, right) after walking each side; check they haven't crossed.
follow-up: Set zeroes in O(1) extra space: use the first row and column as markers (and two flags for themselves).
follow-up: Rotate an image buffer on a microcontroller where you can't allocate a second frame — same trick.

Work on lists of lists of numbers.

* `rotate(mat)` — rotate a **square** matrix 90° clockwise **in place** (return `None`);
* `spiral(mat)` — elements in clockwise spiral order starting top-left (any R×C, including empty);
* `set_zeroes(mat)` — in place: if `mat[i][j] == 0`, set its whole row and column to 0; O(1) extra space
  (don't allocate row/column sets).
"""

def rotate(mat: list[list[int]]) -> None:
    raise NotImplementedError  # TODO: implement


def spiral(mat: list[list[int]]) -> list[int]:
    raise NotImplementedError  # TODO: implement


def set_zeroes(mat: list[list[int]]) -> None:
    raise NotImplementedError  # TODO: implement
