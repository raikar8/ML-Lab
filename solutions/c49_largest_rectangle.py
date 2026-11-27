"""
title: Largest rectangle in a histogram, and maximal rectangle of ones
pattern: Stack
level: hard
time: 35
frequency: high
week: 25
day: 2
tags: monotonic-stack, arrays, matrix
complexity: O(n) for the histogram; O(R·C) for the binary matrix
hint: For each bar, the widest rectangle of its height spans to the nearest strictly lower bar on each side.
hint: Keep a stack of increasing heights; when a lower bar arrives, pop and compute the popped bar's rectangle (right edge = current index, left edge = new stack top). Add a 0 sentinel at the end.
follow-up: Maximal square of ones — simpler DP: side[i][j] = 1 + min(up, left, up-left).
follow-up: Count all-ones submatrices (same stack idea, accumulate counts).

* `largest_rectangle(heights)` — area of the largest axis-aligned rectangle inside the histogram, O(n);
* `maximal_rectangle(matrix)` — `matrix` is a list of strings of `'0'`/`'1'`; area of the largest rectangle
  containing only ones (build a histogram per row).

```
largest_rectangle([2, 1, 5, 6, 2, 3]) -> 10
maximal_rectangle(["10100", "10111", "11111", "10010"]) -> 6
```
"""


def largest_rectangle(heights: list[int]) -> int:
    stack: list[int] = []   # indices with increasing heights
    best = 0
    for i, h in enumerate(heights + [0]):
        while stack and heights[stack[-1]] >= h:
            top = stack.pop()
            left = stack[-1] + 1 if stack else 0
            best = max(best, heights[top] * (i - left))
        stack.append(i)
    return best


def maximal_rectangle(matrix: list[str]) -> int:
    if not matrix:
        return 0
    hist = [0] * len(matrix[0])
    best = 0
    for row in matrix:
        hist = [h + 1 if ch == "1" else 0 for h, ch in zip(hist, row)]
        best = max(best, largest_rectangle(hist))
    return best
