"""
House robber — line and circle (1-D DP)

`values[i] >= 0` is the money in house `i`; you can't take two **adjacent** houses.

* `rob_line(values)` — maximum total for houses on a street;
* `rob_circle(values)` — same, but the houses form a circle (first and last are adjacent).

Empty input → 0. O(1) extra space.

```
rob_line([2, 7, 9, 3, 1])   -> 12   # 2 + 9 + 1
rob_circle([2, 3, 2])       -> 3
rob_circle([1, 2, 3, 1])    -> 4
```
"""

def rob_line(values: list[int]) -> int:
    raise NotImplementedError  # TODO: implement


def rob_circle(values: list[int]) -> int:
    raise NotImplementedError  # TODO: implement
