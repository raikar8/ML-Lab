"""
0/1 knapsack and grid paths — 2-D DP and space reduction

* `knapsack(weights, values, capacity)` — max total value with total weight ≤ capacity, each item at most once;
  O(capacity) memory.
* `knapsack_items(weights, values, capacity)` — `(best_value, sorted list of chosen indices)`; among optimal
  solutions any is fine.
* `grid_paths(grid)` — `grid` is a list of strings with `'.'` free and `'#'` blocked; number of paths from top-left
  to bottom-right moving only right/down (0 if start or end is blocked).
* `min_path_sum(cost)` — minimum sum path top-left → bottom-right moving right/down in a matrix of numbers.
"""

def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    raise NotImplementedError  # TODO: implement


def knapsack_items(weights, values, capacity):
    raise NotImplementedError  # TODO: implement


def grid_paths(grid: list[str]) -> int:
    raise NotImplementedError  # TODO: implement


def min_path_sum(cost: list[list[int]]) -> int:
    raise NotImplementedError  # TODO: implement
