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
    n = len(mat)
    for i in range(n):
        for j in range(i + 1, n):
            mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
    for row in mat:
        row.reverse()


def spiral(mat: list[list[int]]) -> list[int]:
    out: list[int] = []
    if not mat or not mat[0]:
        return out
    top, bottom, left, right = 0, len(mat) - 1, 0, len(mat[0]) - 1
    while top <= bottom and left <= right:
        out += [mat[top][j] for j in range(left, right + 1)]
        out += [mat[i][right] for i in range(top + 1, bottom + 1)]
        if top < bottom:
            out += [mat[bottom][j] for j in range(right - 1, left - 1, -1)]
        if left < right:
            out += [mat[i][left] for i in range(bottom - 1, top, -1)]
        top, bottom, left, right = top + 1, bottom - 1, left + 1, right - 1
    return out


def set_zeroes(mat: list[list[int]]) -> None:
    if not mat:
        return
    R, C = len(mat), len(mat[0])
    row0 = any(mat[0][j] == 0 for j in range(C))
    col0 = any(mat[i][0] == 0 for i in range(R))
    for i in range(1, R):
        for j in range(1, C):
            if mat[i][j] == 0:
                mat[i][0] = mat[0][j] = 0
    for i in range(1, R):
        for j in range(1, C):
            if mat[i][0] == 0 or mat[0][j] == 0:
                mat[i][j] = 0
    if row0:
        for j in range(C):
            mat[0][j] = 0
    if col0:
        for i in range(R):
            mat[i][0] = 0
