"""
Merge intervals, and insert one into a sorted list

Intervals are closed pairs `[start, end]` with `start <= end`; touching intervals (`[1, 3]` and `[3, 5]`)
**overlap** and merge.

* `merge(intervals)` — return the merged, non-overlapping intervals sorted by start.
* `insert(merged, new)` — `merged` is already sorted and non-overlapping; return the result of adding `new`
  in **O(n)** (no re-sorting).

```
merge([[1, 3], [8, 10], [2, 6], [15, 18]]) -> [[1, 6], [8, 10], [15, 18]]
insert([[1, 2], [3, 5], [6, 7], [8, 10]], [4, 8]) -> [[1, 2], [3, 10]]
```
"""

def merge(intervals: list[list[int]]) -> list[list[int]]:
    raise NotImplementedError  # TODO: implement


def insert(merged: list[list[int]], new: list[int]) -> list[list[int]]:
    raise NotImplementedError  # TODO: implement
