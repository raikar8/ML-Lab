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
    raise NotImplementedError  # TODO: implement
