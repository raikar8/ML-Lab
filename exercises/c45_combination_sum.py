"""
title: Combination sum — with reuse, and without reuse when the input has duplicates
pattern: Backtracking
level: medium
time: 30
frequency: high
week: 23
day: 2
tags: backtracking, pruning, dedup
complexity: exponential in the worst case; pruning by sorted order cuts most branches
hint: Sort the candidates; recurse with a start index so each combination is built in non-decreasing order (no permutations).
hint: Stop the loop as soon as a candidate exceeds the remaining target (sorted ⇒ all later ones do too).
follow-up: Count the combinations instead of listing them — this becomes the coin-change "ways" DP.
follow-up: Combination sum with exactly k numbers from 1..9, each used once.

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
