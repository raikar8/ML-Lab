"""
title: Trapping rain water — two pointers, and the container with most water
pattern: Two pointers
level: hard
time: 30
frequency: very high
week: 21
day: 4
tags: two-pointers, arrays, prefix-max
complexity: O(n) time, O(1) space
hint: Water above bar i = min(max height to the left, max to the right) − height[i].
hint: Two pointers: move the side with the smaller running max inward — that side's water level is already decided.
follow-up: 2-D version (trapping rain water on a height map) — a min-heap from the border inward.
follow-up: Container problem: why is moving the shorter line always safe?

* `trap(height)` — total units of water trapped between bars of the given non-negative heights, O(n) time,
  O(1) extra space;
* `max_container(height)` — the largest area `min(h[i], h[j]) × (j − i)` over pairs `i < j` (0 if fewer than
  two bars), O(n).

```
trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])        -> 6
max_container([1, 8, 6, 2, 5, 4, 8, 3, 7])         -> 49
```
"""


def trap(height: list[int]) -> int:
    lo, hi = 0, len(height) - 1
    left_max = right_max = water = 0
    while lo < hi:
        if height[lo] <= height[hi]:
            left_max = max(left_max, height[lo])
            water += left_max - height[lo]
            lo += 1
        else:
            right_max = max(right_max, height[hi])
            water += right_max - height[hi]
            hi -= 1
    return water


def max_container(height: list[int]) -> int:
    lo, hi, best = 0, len(height) - 1, 0
    while lo < hi:
        best = max(best, min(height[lo], height[hi]) * (hi - lo))
        if height[lo] < height[hi]:
            lo += 1
        else:
            hi -= 1
    return best
