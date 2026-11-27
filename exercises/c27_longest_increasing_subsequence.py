"""
Longest increasing subsequence in O(n log n) — length and one witness

* `lis_length(nums)` — length of the longest **strictly** increasing subsequence;
* `lis(nums)` — one such subsequence (as a list of values). Any valid longest one is accepted.

Both O(n log n) (`bisect` allowed).

```
lis_length([10, 9, 2, 5, 3, 7, 101, 18]) -> 4    # e.g. [2, 3, 7, 18]
```
"""

import bisect

def lis(nums: list[int]) -> list[int]:
    raise NotImplementedError  # TODO: implement


def lis_length(nums: list[int]) -> int:
    raise NotImplementedError  # TODO: implement
