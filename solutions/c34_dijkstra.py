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
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [float("inf")] * n
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                if _prev is not None:
                    _prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist


def shortest_path(n: int, edges, src: int, dst: int):
    prev = [-1] * n
    dist = dijkstra(n, edges, src, prev)
    if dist[dst] == float("inf"):
        return float("inf"), []
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return dist[dst], path[::-1]


def network_delay(n: int, edges, src: int):
    worst = max(dijkstra(n, edges, src))
    return -1 if worst == float("inf") else worst
