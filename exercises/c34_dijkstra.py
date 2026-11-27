"""
title: Dijkstra — shortest paths, path reconstruction, and network delay
pattern: Graphs
level: medium
time: 30
frequency: very high
week: 17
day: 4
tags: graph, shortest-path, heap, dijkstra
complexity: O((V + E) log V) time with a binary heap, O(V + E) space
hint: Pop the closest unfinished node from a min-heap; relax its outgoing edges; skip stale heap entries.
hint: Store prev[v] when you improve dist[v]; walk prev back from the target to rebuild the path.
follow-up: Negative edges → Bellman–Ford (detect negative cycles). All-pairs on a small dense graph → Floyd–Warshall.
follow-up: Cheapest path with at most k stops (Bellman–Ford limited to k+1 rounds, or BFS over (node, stops)).

Nodes are `0..n-1`; `edges` holds directed `(u, v, w)` with `w >= 0`.

* `dijkstra(n, edges, src)` — list of shortest distances from `src` (`float("inf")` if unreachable);
* `shortest_path(n, edges, src, dst)` — `(distance, [src, ..., dst])`, or `(inf, [])` if unreachable;
  among equal-length paths any is fine;
* `network_delay(n, edges, src)` — time for a signal from `src` to reach **all** nodes (max distance), or `-1`.
"""

import heapq

def dijkstra(n: int, edges, src: int, _prev=None):
    raise NotImplementedError  # TODO: implement


def shortest_path(n: int, edges, src: int, dst: int):
    raise NotImplementedError  # TODO: implement


def network_delay(n: int, edges, src: int):
    raise NotImplementedError  # TODO: implement
