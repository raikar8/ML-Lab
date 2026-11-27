import functools
import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(a, b):
    @functools.lru_cache(None)
    def go(i, j):
        if i == len(a) or j == len(b):
            return 0
        if a[i] == b[j]:
            return 1 + go(i + 1, j + 1)
        return max(go(i + 1, j), go(i, j + 1))
    return go(0, 0)


def is_subseq(s, t):
    it = iter(t)
    return all(any(x == y for y in it) for x in s)


def test_examples():
    assert m.lcs_length("abcde", "ace") == 3 and m.lcs_length("abc", "def") == 0
    assert m.diff(["a", "b", "c"], ["a", "c", "d"]) == [(" ", "a"), ("-", "b"), (" ", "c"), ("+", "d")]
    assert m.lcs("", "abc") == [] and m.diff([], ["x"]) == [("+", "x")]


def test_random():
    rng = random.Random(35)
    for _ in range(500):
        a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 9)))
        b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 9)))
        n = brute(a, b)
        assert m.lcs_length(a, b) == n
        s = m.lcs(a, b)
        assert len(s) == n and is_subseq(s, a) and is_subseq(s, b)
        d = m.diff(a, b)
        assert "".join(x for t, x in d if t != "+") == a and "".join(x for t, x in d if t != "-") == b
        assert sum(t == " " for t, _ in d) == n
