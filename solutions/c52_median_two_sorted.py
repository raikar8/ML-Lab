"""
title: Median of two sorted arrays in O(log(min(m, n)))
pattern: Binary search
level: hard
time: 40
frequency: high
week: 26
day: 4
tags: binary-search, partition, invariants
complexity: O(log(min(m, n))) time, O(1) space
hint: Partition both arrays so the left parts together hold half the elements; binary-search the cut i in the shorter one (j follows).
hint: The cut is right when max(left parts) ≤ min(right parts). Use ±∞ for empty sides; move i by comparing a[i−1] with b[j].
follow-up: k-th smallest of two sorted arrays (same idea, or discard k/2 elements per step).
follow-up: Median of k sorted arrays (binary search on the value + counting with bisect).

Return the median (float) of the union of two sorted lists `a` and `b` (not both empty). O(log(min(m, n))) —
no merging, no slicing of the arrays inside the loop.

```
median_two([1, 3], [2])    -> 2.0
median_two([1, 2], [3, 4]) -> 2.5
```
"""


def median_two(a: list[int], b: list[int]) -> float:
    if len(a) > len(b):
        a, b = b, a
    m_, n = len(a), len(b)
    half = (m_ + n + 1) // 2
    lo, hi = 0, m_
    INF = float("inf")
    while True:
        i = (lo + hi) // 2
        j = half - i
        a_left = a[i - 1] if i > 0 else -INF
        a_right = a[i] if i < m_ else INF
        b_left = b[j - 1] if j > 0 else -INF
        b_right = b[j] if j < n else INF
        if a_left <= b_right and b_left <= a_right:
            if (m_ + n) % 2:
                return float(max(a_left, b_left))
            return (max(a_left, b_left) + min(a_right, b_right)) / 2
        if a_left > b_right:
            hi = i - 1
        else:
            lo = i + 1
