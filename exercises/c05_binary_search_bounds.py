"""
Binary search done right — lower bound, upper bound, first/last position

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
