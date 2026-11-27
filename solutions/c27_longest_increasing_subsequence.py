"""
title: Longest increasing subsequence in O(n log n) — length and one witness
pattern: Dynamic programming
level: medium
time: 30
frequency: very high
week: 14
day: 2
tags: dp, binary-search, patience-sorting
complexity: O(n log n) time, O(n) space
hint: tails[L] = the smallest possible last value of an increasing subsequence of length L+1. It's sorted.
hint: For each x, binary-search the first tail ≥ x and replace it (or append). Keep predecessor indices to rebuild a witness.
follow-up: Longest non-decreasing subsequence — what single change? (search for the first tail > x)
follow-up: Russian-doll envelopes: sort by width ascending and height descending, then LIS on heights.

* `lis_length(nums)` — length of the longest **strictly** increasing subsequence;
* `lis(nums)` — one such subsequence (as a list of values). Any valid longest one is accepted.

Both O(n log n) (`bisect` allowed).

```
lis_length([10, 9, 2, 5, 3, 7, 101, 18]) -> 4    # e.g. [2, 3, 7, 18]
```
"""
import bisect


def lis(nums: list[int]) -> list[int]:
    tails: list[int] = []      # tail values
    tail_idx: list[int] = []   # index in nums of each tail
    prev = [-1] * len(nums)
    for i, x in enumerate(nums):
        j = bisect.bisect_left(tails, x)
        if j == len(tails):
            tails.append(x)
            tail_idx.append(i)
        else:
            tails[j] = x
            tail_idx[j] = i
        prev[i] = tail_idx[j - 1] if j else -1
    out, k = [], tail_idx[-1] if tail_idx else -1
    while k != -1:
        out.append(nums[k])
        k = prev[k]
    return out[::-1]


def lis_length(nums: list[int]) -> int:
    return len(lis(nums))
