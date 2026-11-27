"""
title: Binary search on the answer — minimum speed, and splitting work across devices
pattern: Binary search
level: medium
time: 30
frequency: very high
week: 22
day: 2
tags: binary-search, monotone-predicate, greedy
complexity: O(n log(max value)) time, O(1) space
hint: "Can we finish at speed v?" is monotone in v — search the smallest v for which a greedy check says yes.
hint: For splitting: given a cap C, greedily fill each device until the next layer would exceed C; count devices.
follow-up: Pipeline-parallel stage balancing is exactly split_min_max on per-layer costs (contiguous stages!).
follow-up: Real-valued answer (e.g. minimum radius): binary search on floats for a fixed number of iterations.

* `min_eating_speed(piles, hours)` — piles of bananas; each hour you pick one pile and eat up to `v` from it.
  Return the minimum integer speed `v >= 1` to finish all piles within `hours` (`hours >= len(piles)`).
* `split_min_max(costs, k)` — split `costs` (positive, in order) into at most `k` **contiguous** groups to
  minimise the largest group sum; return that minimum largest sum. (Think: assign consecutive layers of a model
  to `k` devices.)

Both by binary search over the answer with a greedy feasibility check.

```
min_eating_speed([3, 6, 7, 11], 8) -> 4
split_min_max([7, 2, 5, 10, 8], 2)  -> 18      # [7, 2, 5] | [10, 8]
```
"""

def min_eating_speed(piles: list[int], hours: int) -> int:
    raise NotImplementedError  # TODO: implement


def split_min_max(costs: list[int], k: int) -> int:
    raise NotImplementedError  # TODO: implement
