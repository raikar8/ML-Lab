"""
Top-k frequent elements

Return the `k` most frequent values of `nums`, ordered by **decreasing frequency**; ties go to the value that
**appears first** in `nums`. If there are fewer than `k` distinct values, return them all.
Don't use `Counter.most_common`.

```
top_k_frequent([1, 1, 1, 2, 2, 3], 2) -> [1, 2]
top_k_frequent([4, 5, 5, 4, 6], 2)    -> [4, 5]   # 4 and 5 tie; 4 appears first
```
"""

import heapq

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    raise NotImplementedError  # TODO: implement
