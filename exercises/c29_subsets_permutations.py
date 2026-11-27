"""
Subsets and permutations — with duplicates in the input

Inputs may contain duplicates; outputs must contain **no duplicate** collections.

* `subsets(nums)` — all distinct subsets, each sorted ascending, the whole list sorted (by Python list order);
* `permutations(nums)` — all distinct orderings, sorted lexicographically.

Don't use `itertools`.

```
subsets([1, 2, 2])      -> [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
permutations([1, 1, 2]) -> [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
```
"""

def subsets(nums: list[int]) -> list[list[int]]:
    raise NotImplementedError  # TODO: implement


def permutations(nums: list[int]) -> list[list[int]]:
    raise NotImplementedError  # TODO: implement
