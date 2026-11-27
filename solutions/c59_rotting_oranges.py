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
    R = len(grid)
    C = len(grid[0]) if R else 0
    t = [[-1] * C for _ in range(R)]
    q = deque()
    fresh = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 2:
                t[r][c] = 0
                q.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1
    last = 0
    while q:
        r, c = q.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 1 and t[nr][nc] < 0:
                t[nr][nc] = t[r][c] + 1
                last = t[nr][nc]
                fresh -= 1
                q.append((nr, nc))
    return -1 if fresh else last


def nearest_zero(mat: list[list[int]]) -> list[list[int]]:
    R, C = len(mat), len(mat[0])
    dist = [[-1] * C for _ in range(R)]
    q = deque()
    for r in range(R):
        for c in range(C):
            if mat[r][c] == 0:
                dist[r][c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < R and 0 <= nc < C and dist[nr][nc] < 0:
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return dist
