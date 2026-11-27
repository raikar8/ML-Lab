"""
title: 0/1 knapsack and grid paths — 2-D DP and space reduction
pattern: Dynamic programming
level: medium
time: 35
frequency: high
week: 26
day: 2
tags: dp, knapsack, grid, space-optimisation
complexity: knapsack O(n·W) time, O(W) space; grid O(R·C) time, O(C) space
hint: Knapsack: best[w] over capacities; iterate items outside and capacity DOWNWARD so each item is used at most once.
hint: Grid: paths[r][c] = paths[r−1][c] + paths[r][c−1] (0 on obstacles); min-cost path uses min instead of +.
follow-up: Pick which layers to quantise to 4 bits under a memory budget, maximising accuracy — it's a knapsack.
follow-up: Reconstruct the chosen items (keep the 2-D table, or a "taken" bitset per item).

* `knapsack(weights, values, capacity)` — max total value with total weight ≤ capacity, each item at most once;
  O(capacity) memory.
* `knapsack_items(weights, values, capacity)` — `(best_value, sorted list of chosen indices)`; among optimal
  solutions any is fine.
* `grid_paths(grid)` — `grid` is a list of strings with `'.'` free and `'#'` blocked; number of paths from top-left
  to bottom-right moving only right/down (0 if start or end is blocked).
* `min_path_sum(cost)` — minimum sum path top-left → bottom-right moving right/down in a matrix of numbers.
"""


def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            if best[c - w] + v > best[c]:
                best[c] = best[c - w] + v
    return best[capacity]


def knapsack_items(weights, values, capacity):
    n = len(weights)
    T = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            T[i][c] = T[i - 1][c]
            if w <= c and T[i - 1][c - w] + v > T[i][c]:
                T[i][c] = T[i - 1][c - w] + v
    chosen, c = [], capacity
    for i in range(n, 0, -1):
        if T[i][c] != T[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    return T[n][capacity], sorted(chosen)


def grid_paths(grid: list[str]) -> int:
    if not grid or grid[0][0] == "#" or grid[-1][-1] == "#":
        return 0
    C = len(grid[0])
    row = [0] * C
    row[0] = 1
    for r, line in enumerate(grid):
        for c in range(C):
            if line[c] == "#":
                row[c] = 0
            elif c:
                row[c] += row[c - 1]
    return row[-1]


def min_path_sum(cost: list[list[int]]) -> int:
    C = len(cost[0])
    row = [float("inf")] * C
    row[0] = 0
    for line in cost:
        row[0] += line[0]
        for c in range(1, C):
            row[c] = min(row[c], row[c - 1]) + line[c]
    return row[-1]
