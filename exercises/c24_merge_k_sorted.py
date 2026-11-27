"""
Merge k sorted lists with a heap

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
