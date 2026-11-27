"""
title: k closest points to a query (heap) — the core of brute-force kNN retrieval
pattern: Heap / priority queue
level: medium
time: 20
frequency: very high
week: 16
day: 2
tags: heap, geometry, knn, retrieval
complexity: O(n log k) time, O(k) space
hint: Keep a max-heap of size k on distance (push negative distances into heapq); pop when it grows past k.
hint: Compare squared distances — no sqrt needed for ranking.
follow-up: n is 10⁹ vectors in 768-D: what changes? (ANN: IVF, HNSW, product quantization — accuracy vs latency)
follow-up: Vectorised NumPy version with argpartition: O(n) average.

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
