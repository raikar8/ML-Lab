"""
k closest points to a query (heap) — the core of brute-force kNN retrieval

`points` is a list of tuples of equal dimension. Return the `k` points closest to `query` (Euclidean),
sorted by increasing distance; ties broken by the point's **index** in the input (earlier first).
If `k >= len(points)` return all of them sorted. Use `heapq` with a heap of size at most `k`
(no full sort of all points).

```
k_closest([(1, 3), (-2, 2), (5, 8), (0, 1)], (0, 0), 2) -> [(0, 1), (-2, 2)]
```
"""

import heapq

def k_closest(points: list[tuple], query: tuple, k: int) -> list[tuple]:
    raise NotImplementedError  # TODO: implement
