"""
Sliding-window maximum with a monotonic deque

* `window_max(nums, k)` — list of maxima of each window `nums[i:i+k]`, `i = 0..n−k`; `1 <= k <= len(nums)`;
* `PeakHold(k)` — streaming version: `push(x)` returns the max of the last `min(k, samples so far)` samples.

Both amortised O(1) per element.

```
window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) -> [3, 3, 5, 5, 6, 7]
```
"""

from collections import deque

class PeakHold:
    def __init__(self, k: int):
        raise NotImplementedError  # TODO: implement

    def push(self, x):
        raise NotImplementedError  # TODO: implement


def window_max(nums: list[int], k: int) -> list[int]:
    raise NotImplementedError  # TODO: implement
