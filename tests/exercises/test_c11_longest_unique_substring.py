import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(s):
    best = (0, 0)
    for a in range(len(s)):
        for b in range(a + 1, len(s) + 1):
            if len(set(s[a:b])) == b - a and b - a > best[0]:
                best = (b - a, a)
    return best


def test_examples():
    assert m.longest_unique("abcabcbb") == (3, 0)
    assert m.longest_unique("pwwkew") == (3, 2)
    assert m.longest_unique("") == (0, 0)
    assert m.longest_unique("bbbb") == (1, 0)
    assert m.longest_unique("abba") == (2, 0)     # left must never move backwards


def test_random_against_brute_force():
    rng = random.Random(11)
    for _ in range(1000):
        s = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 14)))
        assert m.longest_unique(s) == brute(s), s


def test_linear():
    s = "".join(chr(97 + (i * 7) % 26) for i in range(400_000))
    assert m.longest_unique(s)[0] == 26
