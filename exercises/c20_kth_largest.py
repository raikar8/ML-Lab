"""
title: k-th largest element — quickselect and heap
pattern: Heap / priority queue
level: medium
time: 25
frequency: very high
week: 10
day: 4
tags: quickselect, heap, partition
complexity: quickselect O(n) average / O(n²) worst; heap O(n log k)
hint: Heap: keep a min-heap of the k largest so far; its root is the answer.
hint: Quickselect: partition around a random pivot into < , ==, > and recurse into the side that contains rank k.
follow-up: Why the random pivot? What input kills a "first element" pivot? (sorted input → O(n²))
follow-up: Median of a stream (two heaps) — see the later problem.

Implement two independent methods (don't sort the whole list, don't use `heapq.nlargest`):

* `kth_largest_heap(nums, k)` — using a size-k min-heap (`heapq.heappush/heappop/heapreplace` allowed);
* `kth_largest_select(nums, k, seed=0)` — iterative or recursive quickselect with a random pivot from
  `random.Random(seed)`; three-way partition so many duplicates stay fast.

`1 <= k <= len(nums)`; don't modify the caller's list.

```
kth_largest_heap([3, 2, 1, 5, 6, 4], 2)        -> 5
kth_largest_select([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) -> 4
```
"""

import heapq

import random

def kth_largest_heap(nums: list[int], k: int) -> int:
    raise NotImplementedError  # TODO: implement


def kth_largest_select(nums: list[int], k: int, seed: int = 0) -> int:
    raise NotImplementedError  # TODO: implement
