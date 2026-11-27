"""
Dijkstra — shortest paths, path reconstruction, and network delay

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
