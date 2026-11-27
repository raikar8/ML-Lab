"""
title: Merge intervals, and insert one into a sorted list
pattern: Intervals
level: medium
time: 25
frequency: very high
week: 5
day: 4
tags: intervals, sorting, sweep
complexity: O(n log n) for merge (sort), O(n) for insert into an already-merged list
hint: Sort by start. An interval overlaps the last merged one iff its start ≤ the last end.
hint: For insert: copy everything ending before the new start, absorb everything that overlaps, copy the rest.
follow-up: Total length covered by a union of intervals; and the gaps between them.
follow-up: Intervals arrive one at a time — keep the merged set queryable (balanced tree / sorted list + bisect).

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
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def insert(merged: list[list[int]], new: list[int]) -> list[list[int]]:
    s, e = new
    out, i, n = [], 0, len(merged)
    while i < n and merged[i][1] < s:
        out.append(list(merged[i]))
        i += 1
    while i < n and merged[i][0] <= e:
        s, e = min(s, merged[i][0]), max(e, merged[i][1])
        i += 1
    out.append([s, e])
    out.extend(list(x) for x in merged[i:])
    return out
