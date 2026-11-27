"""
Maximum-sum subarray (Kadane) — return the sum and the slice

Return `(best_sum, start, end)` for a **non-empty** contiguous subarray `nums[start:end]` with the largest sum.
Ties: prefer the **earliest start**, then the **shortest** subarray. `nums` is non-empty.

```
max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) -> (6, 3, 7)
max_subarray([-3, -1, -2])                    -> (-1, 1, 2)
```
"""

def max_subarray(nums: list[int]) -> tuple[int, int, int]:
    raise NotImplementedError  # TODO: implement
