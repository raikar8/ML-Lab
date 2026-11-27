"""
title: Binary search done right — lower bound, upper bound, first/last position
pattern: Binary search
level: easy
time: 20
frequency: very high
week: 3
day: 2
tags: binary-search, invariants
complexity: O(log n) per query, O(1) space
hint: Keep a half-open window [lo, hi) and an invariant: everything left of lo is < x, everything from hi on is ≥ x.
hint: Loop while lo < hi; mid = (lo + hi) // 2; move lo = mid + 1 or hi = mid — never mid ± 1 on both sides.
follow-up: Generalise to "first index where predicate(i) is True" for any monotone predicate.
follow-up: Search a sorted array of unknown length (only get(i) that raises past the end).

Implement on a **sorted** list `a` (duplicates allowed), without `bisect`:

* `lower_bound(a, x)` — first index `i` with `a[i] >= x` (or `len(a)`),
* `upper_bound(a, x)` — first index `i` with `a[i] > x` (or `len(a)`),
* `search_range(a, x)` — `(first, last)` indices of `x`, or `(-1, -1)` if absent,
* `count(a, x)` — number of occurrences of `x`.

All in O(log n).
"""

def lower_bound(a: list[int], x: int) -> int:
    raise NotImplementedError  # TODO: implement


def upper_bound(a: list[int], x: int) -> int:
    raise NotImplementedError  # TODO: implement


def search_range(a: list[int], x: int) -> tuple[int, int]:
    raise NotImplementedError  # TODO: implement


def count(a: list[int], x: int) -> int:
    raise NotImplementedError  # TODO: implement
