"""
Bit manipulation essentials

Implement without converting to strings (no `bin()`, `format`, `str`) and without `int.bit_count`:

* `popcount(x)` — number of set bits of a non-negative int;
* `is_power_of_two(x)` — True for 1, 2, 4, …; False for 0 and negatives;
* `single_number(nums)` — every element appears twice except one; return it in O(n) time, O(1) space;
* `two_singles(nums)` — every element appears twice except two; return them as a sorted tuple;
* `reverse_bits32(x)` — reverse the 32 bits of `0 <= x < 2**32`;
* `count_bits_upto(n)` — list of popcounts for 0..n, O(n).
"""

def popcount(x: int) -> int:
    raise NotImplementedError  # TODO: implement


def is_power_of_two(x: int) -> bool:
    raise NotImplementedError  # TODO: implement


def single_number(nums: list[int]) -> int:
    raise NotImplementedError  # TODO: implement


def two_singles(nums: list[int]) -> tuple[int, int]:
    raise NotImplementedError  # TODO: implement


def reverse_bits32(x: int) -> int:
    raise NotImplementedError  # TODO: implement


def count_bits_upto(n: int) -> list[int]:
    raise NotImplementedError  # TODO: implement
