"""
title: House robber — line and circle (1-D DP)
pattern: Dynamic programming
level: medium
time: 20
frequency: very high
week: 8
day: 4
tags: dp, state-machine
complexity: O(n) time, O(1) space
hint: best(i) = max(best(i−1), best(i−2) + v[i]) — either skip house i or take it.
hint: Circle: the first and last houses can't both be taken → answer = max(line without first, line without last).
follow-up: Return which houses to rob, not just the amount (keep choices, then backtrack).
follow-up: Houses on a binary tree (return (take, skip) pairs from each subtree).

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
