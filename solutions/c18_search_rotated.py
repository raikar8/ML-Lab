"""
title: Search in a rotated sorted array, and find the rotation point
pattern: Binary search
level: medium
time: 25
frequency: very high
week: 9
day: 4
tags: binary-search, invariants
complexity: O(log n) time, O(1) space
hint: At any mid, at least one half — [lo, mid] or [mid, hi] — is sorted. Check whether the target lies in that half.
hint: Minimum: compare a[mid] with a[hi]; if a[mid] > a[hi] the minimum is right of mid, else at mid or left.
follow-up: With duplicates the worst case becomes O(n) — why, and which comparison breaks?
follow-up: Find a peak element (any i with a[i] > neighbours) in O(log n).

A sorted array of **distinct** integers was rotated at an unknown pivot (e.g. `[4,5,6,7,0,1,2]`).

* `find_min_index(a)` — index of the smallest element (0 if not rotated); `a` is non-empty;
* `search(a, target)` — index of `target` or `-1`.

Both in O(log n), no linear scans.
"""


def find_min_index(a: list[int]) -> int:
    lo, hi = 0, len(a) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] > a[hi]:
            lo = mid + 1
        else:
            hi = mid
    return lo


def search(a: list[int], target: int) -> int:
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == target:
            return mid
        if a[lo] <= a[mid]:                 # left half sorted
            if a[lo] <= target < a[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                               # right half sorted
            if a[mid] < target <= a[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
