"""
title: Greedy — jump game (reachability and fewest jumps) and gas station
pattern: Greedy
level: medium
time: 30
frequency: high
week: 25
day: 4
tags: greedy, arrays, proof
complexity: O(n) time, O(1) space each
hint: Jump game: track the farthest index reachable so far; if i ever exceeds it you're stuck. Fewest jumps = BFS by "levels" of reach.
hint: Gas station: if the total gas ≥ total cost a solution exists; whenever the running tank goes negative, the start must be after this point.
follow-up: Prove the gas-station greedy (if you can't reach j from i, no start between i and j works either).
follow-up: When does greedy fail and you need DP? (coin change with arbitrary denominations, jump game with costs)

* `can_reach_end(jumps)` — `jumps[i]` is the maximum jump length from index `i`; can you reach the last index
  from index 0?
* `min_jumps(jumps)` — fewest jumps to reach the last index, or `-1` if impossible (0 for a single element);
* `gas_start(gas, cost)` — a circular route; at station `i` you add `gas[i]` and driving to `i+1` costs `cost[i]`.
  Return the **unique smallest** start index from which you can complete the loop with a tank starting empty, or
  `-1`.
"""


def can_reach_end(jumps: list[int]) -> bool:
    far = 0
    for i, j in enumerate(jumps):
        if i > far:
            return False
        far = max(far, i + j)
    return True


def min_jumps(jumps: list[int]) -> int:
    n = len(jumps)
    steps, cur_end, far = 0, 0, 0
    for i in range(n - 1):
        far = max(far, i + jumps[i])
        if i == cur_end:
            if far <= i:
                return -1
            steps += 1
            cur_end = far
            if cur_end >= n - 1:
                break
    return steps


def gas_start(gas: list[int], cost: list[int]) -> int:
    if sum(gas) < sum(cost):
        return -1
    start, tank = 0, 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start, tank = i + 1, 0
    return start
