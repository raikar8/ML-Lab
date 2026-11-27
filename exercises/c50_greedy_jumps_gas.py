"""
Greedy — jump game (reachability and fewest jumps) and gas station

* `can_reach_end(jumps)` — `jumps[i]` is the maximum jump length from index `i`; can you reach the last index
  from index 0?
* `min_jumps(jumps)` — fewest jumps to reach the last index, or `-1` if impossible (0 for a single element);
* `gas_start(gas, cost)` — a circular route; at station `i` you add `gas[i]` and driving to `i+1` costs `cost[i]`.
  Return the **unique smallest** start index from which you can complete the loop with a tank starting empty, or
  `-1`.
"""

def can_reach_end(jumps: list[int]) -> bool:
    raise NotImplementedError  # TODO: implement


def min_jumps(jumps: list[int]) -> int:
    raise NotImplementedError  # TODO: implement


def gas_start(gas: list[int], cost: list[int]) -> int:
    raise NotImplementedError  # TODO: implement
