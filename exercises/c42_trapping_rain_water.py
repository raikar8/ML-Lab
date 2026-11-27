"""
Trapping rain water — two pointers, and the container with most water

* `trap(height)` — total units of water trapped between bars of the given non-negative heights, O(n) time,
  O(1) extra space;
* `max_container(height)` — the largest area `min(h[i], h[j]) × (j − i)` over pairs `i < j` (0 if fewer than
  two bars), O(n).

```
trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])        -> 6
max_container([1, 8, 6, 2, 5, 4, 8, 3, 7])         -> 49
```
"""

def trap(height: list[int]) -> int:
    raise NotImplementedError  # TODO: implement


def max_container(height: list[int]) -> int:
    raise NotImplementedError  # TODO: implement
