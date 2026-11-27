"""
title: Bit manipulation essentials
pattern: Bit manipulation & math
level: easy
time: 25
frequency: high
week: 21
day: 2
tags: bits, xor, embedded
complexity: O(1) or O(number of bits) per call; O(n) for the array problems
hint: x & (x − 1) clears the lowest set bit; x & −x isolates it. a ^ a = 0 and a ^ 0 = a.
hint: For the "two singles" problem: XOR everything, then split the numbers by any bit where the two singles differ.
follow-up: Reverse the bits of a 32-bit word in O(log 32) steps with masks (classic DSP / FFT bit-reversal).
follow-up: Count set bits for all numbers 0..n in O(n) with dp[i] = dp[i >> 1] + (i & 1).

Implement without converting to strings (no `bin()`, `format`, `str`) and without `int.bit_count`:

* `popcount(x)` — number of set bits of a non-negative int;
* `is_power_of_two(x)` — True for 1, 2, 4, …; False for 0 and negatives;
* `single_number(nums)` — every element appears twice except one; return it in O(n) time, O(1) space;
* `two_singles(nums)` — every element appears twice except two; return them as a sorted tuple;
* `reverse_bits32(x)` — reverse the 32 bits of `0 <= x < 2**32`;
* `count_bits_upto(n)` — list of popcounts for 0..n, O(n).
"""


def popcount(x: int) -> int:
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def is_power_of_two(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


def single_number(nums: list[int]) -> int:
    acc = 0
    for v in nums:
        acc ^= v
    return acc


def two_singles(nums: list[int]) -> tuple[int, int]:
    both = single_number(nums)
    low = both & -both          # a bit where the two singles differ
    a = 0
    for v in nums:
        if v & low:
            a ^= v
    b = both ^ a
    return (a, b) if a <= b else (b, a)


def reverse_bits32(x: int) -> int:
    x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1)
    x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2)
    x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4)
    x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8)
    x = ((x >> 16) | (x << 16)) & 0xFFFFFFFF
    return x


def count_bits_upto(n: int) -> list[int]:
    out = [0] * (n + 1)
    for i in range(1, n + 1):
        out[i] = out[i >> 1] + (i & 1)
    return out
