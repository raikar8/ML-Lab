"""
title: Product of everything except self — no division
pattern: Arrays & hashing
level: medium
time: 20
frequency: very high
week: 4
day: 4
tags: prefix-products, arrays
complexity: O(n) time, O(1) extra space besides the output
hint: out[i] = (product of everything left of i) × (product of everything right of i).
hint: Fill out[] with left products in one pass, then sweep from the right with a running product.
follow-up: Why is "total product / nums[i]" a bad idea even when division is allowed? (zeros, overflow)
follow-up: Same idea for sums: return, for each i, the sum of the k elements around it.

Return `out` where `out[i]` is the product of all elements of `nums` except `nums[i]`.
**Don't use division**, and use O(1) extra space apart from the output list. Zeros are allowed.

```
product_except_self([1, 2, 3, 4]) -> [24, 12, 8, 6]
product_except_self([0, 4, 0])    -> [0, 0, 0]
```
"""


def product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    out = [1] * n
    left = 1
    for i in range(n):
        out[i] = left
        left *= nums[i]
    right = 1
    for i in range(n - 1, -1, -1):
        out[i] *= right
        right *= nums[i]
    return out
