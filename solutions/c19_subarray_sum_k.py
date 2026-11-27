"""
title: Count subarrays with sum k (prefix sums + hash map)
pattern: Arrays & hashing
level: medium
time: 25
frequency: very high
week: 10
day: 2
tags: prefix-sums, hash-map
complexity: O(n) time, O(n) space
hint: sum(a[i:j]) = P[j] − P[i] with prefix sums P. You want pairs i < j with P[i] = P[j] − k.
hint: Sweep j, keep counts of prefix sums seen so far (start with {0: 1}). Sliding windows fail with negatives.
follow-up: Longest subarray with sum k (store the first index of each prefix sum instead of counts).
follow-up: Count subarrays whose sum is divisible by k (prefix sums mod k).

* `count_subarrays(nums, k)` — number of contiguous non-empty subarrays whose sum equals `k`
  (values may be negative);
* `longest_subarray(nums, k)` — length of the longest such subarray, 0 if none.

Both O(n).

```
count_subarrays([1, 1, 1], 2)       -> 2
count_subarrays([1, -1, 1, -1], 0)  -> 4
longest_subarray([1, -1, 5, -2, 3], 3) -> 4
```
"""


def count_subarrays(nums: list[int], k: int) -> int:
    seen = {0: 1}
    total = prefix = 0
    for x in nums:
        prefix += x
        total += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return total


def longest_subarray(nums: list[int], k: int) -> int:
    first = {0: -1}
    best = prefix = 0
    for j, x in enumerate(nums):
        prefix += x
        i = first.get(prefix - k)
        if i is not None:
            best = max(best, j - i)
        first.setdefault(prefix, j)
    return best
