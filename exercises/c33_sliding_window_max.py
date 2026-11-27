"""
title: Sliding-window maximum with a monotonic deque
pattern: Sliding window
level: hard
time: 25
frequency: very high
week: 17
day: 2
tags: deque, monotonic-queue, streaming, dsp
complexity: O(n) time, O(k) space
hint: Keep a deque of indices with decreasing values; anything smaller than the incoming value can never be a maximum again.
hint: Drop the front when it leaves the window (index ≤ i − k); the front is the window max.
follow-up: Sliding-window median (two heaps with lazy deletion, or a sorted container).
follow-up: In audio, this is a peak-hold / envelope follower — how would you make it O(1) per sample in C with a fixed ring buffer?

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
