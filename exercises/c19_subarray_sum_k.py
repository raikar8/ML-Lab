"""
Count subarrays with sum k (prefix sums + hash map)

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
    raise NotImplementedError  # TODO: implement


def longest_subarray(nums: list[int], k: int) -> int:
    raise NotImplementedError  # TODO: implement
