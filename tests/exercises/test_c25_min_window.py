import random
from collections import Counter

from helpers import key_of, load

m = load(key_of(__file__))


def brute(s, t):
    need = Counter(t)
    for L in range(1, len(s) + 1):
        for i in range(len(s) - L + 1):
            c = Counter(s[i:i + L])
            if all(c[k] >= v for k, v in need.items()):
                return s[i:i + L]
    return ""


def test_examples():
    assert m.min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert m.min_window("a", "aa") == "" and m.min_window("a", "a") == "a"
    assert m.min_window("aab", "aab") == "aab" and m.min_window("baab", "ab") == "ba"


def test_random_against_brute_force():
    rng = random.Random(25)
    for _ in range(600):
        s = "".join(rng.choice("abc") for _ in range(rng.randint(0, 12)))
        t = "".join(rng.choice("abc") for _ in range(rng.randint(1, 4)))
        assert m.min_window(s, t) == brute(s, t), (s, t)


def test_linear():
    s = "x" * 300_000 + "abc" + "y" * 300_000
    assert m.min_window(s, "cba") == "abc"
