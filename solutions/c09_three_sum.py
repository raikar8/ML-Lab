"""
title: 3Sum — all unique triplets that sum to zero
pattern: Two pointers
level: medium
time: 30
frequency: very high
week: 5
day: 2
tags: two-pointers, sorting, dedup
complexity: O(n²) time, O(1) extra space besides the output (after sorting)
hint: Sort. Fix the first element a[i]; the rest is a two-sum on the sorted suffix with two pointers.
hint: Skip duplicate values for the fixed element and after each found triplet, or you'll emit repeats.
follow-up: k-Sum in general (recursion down to two-sum): complexity?
follow-up: 3Sum closest to a target.

Return every **distinct** triplet of values `[a, b, c]` from `nums` (using three different positions) with
`a + b + c == target` (default 0). Each triplet sorted ascending; the list of triplets sorted ascending.

```
three_sum([-1, 0, 1, 2, -1, -4]) -> [[-1, -1, 2], [-1, 0, 1]]
three_sum([0, 0, 0, 0])          -> [[0, 0, 0]]
```
Target: O(n²).
"""


def three_sum(nums: list[int], target: int = 0) -> list[list[int]]:
    a = sorted(nums)
    n, out = len(a), []
    for i in range(n - 2):
        if i and a[i] == a[i - 1]:
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = a[i] + a[lo] + a[hi]
            if s < target:
                lo += 1
            elif s > target:
                hi -= 1
            else:
                out.append([a[i], a[lo], a[hi]])
                lo += 1
                while lo < hi and a[lo] == a[lo - 1]:
                    lo += 1
                hi -= 1
    return out
