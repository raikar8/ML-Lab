import ast
import inspect
import random

from helpers import key_of, load

m = load(key_of(__file__))


def test_basics():
    assert [m.popcount(x) for x in (0, 1, 7, 8, 255, 2**40 + 3)] == [0, 1, 3, 1, 8, 3]
    assert [m.is_power_of_two(x) for x in (0, 1, 2, 3, 64, -4, 2**50)] == [False, True, True, False, True, False, True]
    assert m.single_number([4, 1, 2, 1, 2]) == 4 and m.single_number([-3]) == -3
    assert m.two_singles([1, 2, 1, 3, 2, 5]) == (3, 5)
    assert m.reverse_bits32(1) == 2**31 and m.reverse_bits32(0b1011) == 0b1101 << 28
    assert m.count_bits_upto(5) == [0, 1, 1, 2, 1, 2]


def test_random():
    rng = random.Random(43)
    for _ in range(500):
        x = rng.getrandbits(32)
        assert m.popcount(x) == bin(x).count("1")
        assert m.reverse_bits32(x) == int(format(x, "032b")[::-1], 2)
        pairs = rng.sample(range(-50, 50), 6)
        a, b = pairs[0], pairs[1]
        nums = pairs[2:] * 2 + [a, b]
        rng.shuffle(nums)
        assert m.two_singles(nums) == tuple(sorted((a, b)))
    assert m.count_bits_upto(1000) == [bin(i).count("1") for i in range(1001)]


def test_no_string_tricks():
    tree = ast.parse(inspect.getsource(m))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"bin", "format", "str", "bit_count"}
