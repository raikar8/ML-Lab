"""
title: Maximum-sum subarray (Kadane) — return the sum and the slice
pattern: Dynamic programming
level: easy
time: 20
frequency: very high
week: 2
day: 4
tags: dp, kadane, arrays
complexity: O(n) time, O(1) space
hint: Best subarray ending at i is either nums[i] alone or nums[i] appended to the best ending at i−1.
hint: Track where the current run started, and update the answer whenever the current sum beats it.
follow-up: Circular array (the subarray may wrap around): max(Kadane, total − min-subarray), careful when all are negative.
follow-up: Maximum product subarray — what extra state do you need?

Return `(best_sum, start, end)` for a **non-empty** contiguous subarray `nums[start:end]` with the largest sum.
Ties: prefer the **earliest start**, then the **shortest** subarray. `nums` is non-empty.

```
max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) -> (6, 3, 7)
max_subarray([-3, -1, -2])                    -> (-1, 1, 2)
```
"""


def max_subarray(nums: list[int]) -> tuple[int, int, int]:
    best = (nums[0], 0, 1)
    cur, cur_start = nums[0], 0
    for i in range(1, len(nums)):
        if cur < 0:                      # a negative prefix never helps; restart (keep it on 0 for earliest start)
            cur, cur_start = nums[i], i
        else:
            cur += nums[i]
        s, a, b = best
        if cur > s or (cur == s and (cur_start, i + 1 - cur_start) < (a, b - a)):
            best = (cur, cur_start, i + 1)
    return best
