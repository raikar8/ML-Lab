"""
Running median of a stream with two heaps

`MedianFinder` with `add(x)`, `median()` (float; mean of the two middle values when the count is even; raise
`ValueError` when empty) and `__len__`. Each `add` must be O(log n) — use `heapq`.

```
mf = MedianFinder(); mf.add(1); mf.add(2); mf.median() -> 1.5; mf.add(3); mf.median() -> 2.0
```
"""

import heapq

class MedianFinder:
    def __init__(self):
        raise NotImplementedError  # TODO: implement

    def add(self, x) -> None:
        raise NotImplementedError  # TODO: implement

    def median(self) -> float:
        raise NotImplementedError  # TODO: implement

    def __len__(self) -> int:
        raise NotImplementedError  # TODO: implement
