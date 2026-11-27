"""
Binary search on the answer — minimum speed, and splitting work across devices

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
