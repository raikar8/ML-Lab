import functools
import math
import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(a, b):
    @functools.lru_cache(None)
    def go(i, j):
        if i == len(a):
            return len(b) - j
        if j == len(b):
            return len(a) - i
        return min(go(i + 1, j + 1) + (a[i] != b[j]), go(i + 1, j) + 1, go(i, j + 1) + 1)
    return go(0, 0)


def apply_ops(ops):
    a = [x for o, x, _ in ops if o != "I"]
    b = [y for o, _, y in ops if o != "D"]
    return a, b


def test_examples():
    assert m.edit_distance("kitten", "sitting") == 3
    assert m.edit_distance("", "abc") == 3 and m.edit_distance("same", "same") == 0
    d, ops = m.align("ab", "b")
    assert d == 1 and ops == [("D", "a", None), ("=", "b", "b")]
    w, c = m.wer("the cat sat on the mat", "the cat sit on mat")
    assert math.isclose(w, 2 / 6) and c == {"S": 1, "D": 1, "I": 0, "N": 6}
    assert m.wer("", "")[0] == 0.0 and m.wer("", "hello")[0] == float("inf")
    w, c = m.wer("a b", "a x b y z")
    assert c["I"] == 3 and math.isclose(w, 1.5)


def test_random_alignment_is_valid_and_minimal():
    rng = random.Random(32)
    for _ in range(500):
        a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 8)))
        b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 8)))
        exp = brute(a, b)
        assert m.edit_distance(a, b) == exp
        d, ops = m.align(a, b)
        assert d == exp == sum(o != "=" for o, _, _ in ops)
        ra, rb = apply_ops(ops)
        assert "".join(ra) == a and "".join(rb) == b
        assert all((o == "=") == (x == y) for o, x, y in ops if o in "=S")
