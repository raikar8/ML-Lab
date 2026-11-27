import random
from collections import Counter

from helpers import key_of, load

m = load(key_of(__file__))


def brute(words):
    out = []
    for w in words:
        for g in out:
            if Counter(g[0]) == Counter(w):
                g.append(w)
                break
        else:
            out.append([w])
    return out


def test_examples():
    assert m.group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
    assert m.group_anagrams([]) == []
    assert m.group_anagrams(["", ""]) == [["", ""]]
    assert m.group_anagrams(["ab", "Ba", "ba"]) == [["ab", "ba"], ["Ba"]]
    assert m.group_anagrams(["aab", "abb"]) == [["aab"], ["abb"]]   # multiplicities matter


def test_random_against_brute_force():
    rng = random.Random(2)
    for _ in range(300):
        words = ["".join(rng.choice("abc") for _ in range(rng.randint(0, 4))) for _ in range(rng.randint(0, 15))]
        assert m.group_anagrams(words) == brute(words)
