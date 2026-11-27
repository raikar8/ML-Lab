"""
title: Subsets and permutations — with duplicates in the input
pattern: Backtracking
level: medium
time: 30
frequency: very high
week: 15
day: 2
tags: backtracking, recursion, dedup
complexity: O(n · 2ⁿ) for subsets, O(n · n!) for permutations
hint: Subsets: at position i, choose to include nums[i] or not; or extend the current subset by each later element.
hint: Duplicates: sort first; at the same recursion depth, skip a value equal to the previous one you tried.
follow-up: Generate subsets iteratively with a bitmask; and the k-th permutation directly (factorial number system).
follow-up: Combinations of size k from n — and how to prune when the remaining elements can't fill the slots.

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
    a = sorted(nums)
    out: list[list[int]] = []
    cur: list[int] = []

    def go(start):
        out.append(list(cur))
        for i in range(start, len(a)):
            if i > start and a[i] == a[i - 1]:
                continue
            cur.append(a[i])
            go(i + 1)
            cur.pop()

    go(0)
    return sorted(out)


def permutations(nums: list[int]) -> list[list[int]]:
    a = sorted(nums)
    used = [False] * len(a)
    out: list[list[int]] = []
    cur: list[int] = []

    def go():
        if len(cur) == len(a):
            out.append(list(cur))
            return
        for i in range(len(a)):
            if used[i] or (i and a[i] == a[i - 1] and not used[i - 1]):
                continue
            used[i] = True
            cur.append(a[i])
            go()
            cur.pop()
            used[i] = False

    go()
    return out
