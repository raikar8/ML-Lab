import random
import time

from helpers import key_of, load

m = load(key_of(__file__))


def brute(s, words):
    if s == "":
        return [""]
    out = []
    for w in set(words):
        if w and s.startswith(w):
            out += [(w + " " + r).strip() for r in brute(s[len(w):], words)]
    return sorted(set(out))


def test_examples():
    assert m.can_break("leetcode", ["leet", "code"]) and not m.can_break("catsandog", ["cats", "dog", "sand", "and", "cat"])
    assert m.all_breaks("catsanddog", ["cat", "cats", "and", "sand", "dog"]) == ["cat sand dog", "cats and dog"]
    assert m.all_breaks("", ["a"]) == [""] and m.can_break("", [])


def test_random_against_brute_force():
    rng = random.Random(38)
    for _ in range(300):
        words = list({"".join(rng.choice("ab") for _ in range(rng.randint(1, 3))) for _ in range(rng.randint(1, 5))})
        s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 9)))
        exp = brute(s, words)
        assert m.all_breaks(s, words) == exp and m.can_break(s, words) == bool(exp)


def test_no_blowup_when_impossible():
    t0 = time.perf_counter()
    assert m.all_breaks("a" * 40 + "b", ["a", "aa", "aaa"]) == []
    assert not m.can_break("a" * 5000 + "b", ["a", "aa", "aaa"])
    assert time.perf_counter() - t0 < 2
