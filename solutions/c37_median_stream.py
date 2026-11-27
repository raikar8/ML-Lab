"""
title: Running median of a stream with two heaps
pattern: Heap / priority queue
level: hard
time: 30
frequency: very high
week: 19
day: 2
tags: heap, streaming, two-heaps
complexity: O(log n) per insert, O(1) per median query
hint: A max-heap holds the smaller half and a min-heap the larger half; keep sizes equal or the low side one bigger.
hint: Insert into the low side, move its max to the high side, then rebalance if the high side got bigger.
follow-up: Sliding-window median: add lazy deletion (a dict of "to delete" counts, cleaned when they reach a top).
follow-up: Approximate quantiles for huge streams (t-digest, P² algorithm, reservoir sampling).

`MedianFinder` with `add(x)`, `median()` (float; mean of the two middle values when the count is even; raise
`ValueError` when empty) and `__len__`. Each `add` must be O(log n) — use `heapq`.

```
mf = MedianFinder(); mf.add(1); mf.add(2); mf.median() -> 1.5; mf.add(3); mf.median() -> 2.0
```
"""
import heapq


class MedianFinder:
    def __init__(self):
        self.low: list[float] = []   # max-heap via negation
        self.high: list[float] = []  # min-heap

    def add(self, x) -> None:
        heapq.heappush(self.low, -x)
        heapq.heappush(self.high, -heapq.heappop(self.low))
        if len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self) -> float:
        if not self.low:
            raise ValueError("median of empty stream")
        if len(self.low) > len(self.high):
            return float(-self.low[0])
        return (-self.low[0] + self.high[0]) / 2

    def __len__(self) -> int:
        return len(self.low) + len(self.high)
