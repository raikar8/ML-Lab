"""
title: Course schedule — topological order or detect a cycle
pattern: Graphs
level: medium
time: 30
frequency: very high
week: 11
day: 2
tags: graph, topological-sort, kahn, cycle-detection
complexity: O(V + E) time and space
hint: Kahn's algorithm: repeatedly take a course with no remaining prerequisites (in-degree 0).
hint: If you can't take all V courses, the leftover ones lie on (or behind) a cycle.
follow-up: Return one actual cycle when there is one (DFS with colours and a parent stack).
follow-up: Minimum number of semesters if you can take any number of courses per semester (BFS by levels).

There are `n` courses `0..n-1`; `prereqs` holds pairs `(a, b)` meaning **b must be taken before a**.

* `course_order(n, prereqs)` — a valid order of all courses, or `[]` if impossible. When several courses are
  available, take the **smallest number first** (so the answer is unique — use a heap).
* `can_finish(n, prereqs) -> bool`.

```
course_order(4, [(1, 0), (2, 0), (3, 1), (3, 2)]) -> [0, 1, 2, 3]
course_order(2, [(0, 1), (1, 0)])                 -> []
```
"""
import heapq


def course_order(n: int, prereqs: list[tuple[int, int]]) -> list[int]:
    out_edges: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for a, b in prereqs:
        out_edges[b].append(a)
        indeg[a] += 1
    ready = [c for c in range(n) if indeg[c] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        c = heapq.heappop(ready)
        order.append(c)
        for nxt in out_edges[c]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(ready, nxt)
    return order if len(order) == n else []


def can_finish(n: int, prereqs: list[tuple[int, int]]) -> bool:
    return len(course_order(n, prereqs)) == n
