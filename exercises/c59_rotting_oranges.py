"""
title: Multi-source BFS — rotting oranges and distance to the nearest zero
pattern: Graphs
level: medium
time: 25
frequency: high
week: 23
day: 5
tags: bfs, multi-source, grid
complexity: O(R·C) time and space
hint: Put every source in the queue at time 0 and run one BFS; the BFS layer is the time / distance.
hint: For "rotting", count fresh cells up front; if some are left after the BFS, return −1.
follow-up: Walls and gates / distance from every cell to the nearest charger — the same pattern.
follow-up: Weighted version (different spread times) → Dijkstra from multiple sources.

* `minutes_to_rot(grid)` — `grid` rows of `0` empty, `1` fresh, `2` rotten. Each minute, fresh oranges
  4-adjacent to a rotten one become rotten. Return the minutes until no fresh orange remains, `-1` if impossible,
  `0` if there are no fresh oranges.
* `nearest_zero(mat)` — matrix of 0/1; return a matrix with the 4-neighbour distance from each cell to the
  nearest 0 (there is at least one 0).

Don't modify the inputs.
"""

from collections import deque

def minutes_to_rot(grid: list[list[int]]) -> int:
    raise NotImplementedError  # TODO: implement


def nearest_zero(mat: list[list[int]]) -> list[list[int]]:
    raise NotImplementedError  # TODO: implement
