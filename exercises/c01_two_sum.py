"""
title: Two Sum — indices of the pair that adds up to a target
pattern: Arrays & hashing
level: easy
time: 15
frequency: very high
week: 1
day: 2
tags: hash-map, warm-up
complexity: O(n) time, O(n) space
hint: For each element x you need to know whether target − x appeared earlier — and where.
hint: Keep a dict value → index of what you've seen; check before inserting so an element can't pair with itself.
follow-up: The array is sorted — solve it in O(1) extra space (two pointers from both ends).
follow-up: Return all distinct value pairs instead of one pair of indices; how do you handle duplicates?

Given a list of integers `nums` and an integer `target`, return indices `(i, j)` with `i < j` and
`nums[i] + nums[j] == target`. If several pairs work, return the one whose **second index `j` is smallest**
(and for that `j`, the smallest `i`). Return `None` if no pair exists.

```
two_sum([3, 1, 4, 1, 5], 2)   -> (1, 3)
two_sum([2, 7, 11, 15], 9)    -> (0, 1)
two_sum([1, 2], 7)            -> None
```
Target: one pass, O(n).
"""

def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    raise NotImplementedError  # TODO: implement
