"""
title: Next greater element — days until a warmer temperature
pattern: Stack
level: medium
time: 20
frequency: very high
week: 8
day: 2
tags: monotonic-stack, arrays
complexity: O(n) time (each index pushed and popped once), O(n) space
hint: Keep a stack of indices whose answer is still unknown; their temperatures are non-increasing.
hint: When a warmer day arrives, pop every colder index from the top and fill in its answer.
follow-up: Circular array version (loop over 2n indices).
follow-up: Sum of minimums of all subarrays — previous/next smaller with a monotonic stack.

* `days_until_warmer(t)` — for each day `i`, the number of days until a **strictly** warmer day, or 0 if none.
* `next_greater(nums)` — for each `i`, the value of the next strictly greater element to the right, or `-1`.

Both in O(n).

```
days_until_warmer([73, 74, 75, 71, 69, 72, 76, 73]) -> [1, 1, 4, 2, 1, 1, 0, 0]
next_greater([2, 1, 2, 4, 3])                      -> [4, 2, 4, -1, -1]
```
"""


def days_until_warmer(t: list[int]) -> list[int]:
    out = [0] * len(t)
    stack: list[int] = []
    for i, x in enumerate(t):
        while stack and t[stack[-1]] < x:
            j = stack.pop()
            out[j] = i - j
        stack.append(i)
    return out


def next_greater(nums: list[int]) -> list[int]:
    out = [-1] * len(nums)
    stack: list[int] = []
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            out[stack.pop()] = x
        stack.append(i)
    return out
