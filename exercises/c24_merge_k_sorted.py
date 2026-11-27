"""
title: Merge k sorted lists with a heap
pattern: Heap / priority queue
level: hard
time: 25
frequency: very high
week: 12
day: 4
tags: heap, merge, linked-list
complexity: O(N log k) time for N total elements, O(k) heap space
hint: The next output is the smallest among the current heads of the k lists — a min-heap of size k.
hint: Push (value, list_id, position) so ties never compare the payloads; pop, emit, push that list's next element.
follow-up: Divide and conquer (merge pairs, then pairs of pairs) — same complexity, no heap.
follow-up: External sort: merge k sorted files that don't fit in memory.

* `merge_k(lists)` — merge `k` ascending Python lists into one ascending list;
* `merge_k_iters(iterables)` — same, but the inputs are **iterators** (possibly infinite generators); yield
  results lazily, reading at most one element ahead from each input. Equal values keep the order of the input
  they came from (stable by input index).

Don't use `heapq.merge` or sort the concatenation.
"""

import heapq

def merge_k_iters(iterables):
    raise NotImplementedError  # TODO: implement


def merge_k(lists: list[list[int]]) -> list[int]:
    raise NotImplementedError  # TODO: implement
