"""
Combination sum — with reuse, and without reuse when the input has duplicates

* `combination_sum(candidates, target)` — distinct positive candidates, each usable **any number of times**;
  return all combinations summing to `target`, each sorted ascending, the list sorted.
* `combination_sum_once(candidates, target)` — candidates may repeat; each **position** usable at most once;
  no duplicate combinations in the output; same ordering rules.

```
combination_sum([2, 3, 6, 7], 7)               -> [[2, 2, 3], [7]]
combination_sum_once([10, 1, 2, 7, 6, 1, 5], 8) -> [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]
```
"""

def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    raise NotImplementedError  # TODO: implement


def combination_sum_once(candidates: list[int], target: int) -> list[list[int]]:
    raise NotImplementedError  # TODO: implement
