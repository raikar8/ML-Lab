import random
from collections import Counter

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    t = m.Trie()
    for w in ["apple", "app", "apply", "ape", "bat", "app"]:
        t.insert(w)
    assert t.contains("app") and not t.contains("ap")
    assert t.count_prefix("ap") == 5 and t.count_prefix("") == 6 and t.count_prefix("z") == 0
    assert t.complete("ap", 3) == ["ape", "app", "apple"]
    assert t.remove("app") and t.count_prefix("app") == 3 and t.contains("app")
    assert t.remove("app") and not t.contains("app") and t.contains("apple")
    assert not t.remove("app") and not t.remove("zzz")


def test_random_against_counter():
    rng = random.Random(26)
    t, ref = m.Trie(), Counter()
    for _ in range(3000):
        w = "".join(rng.choice("abc") for _ in range(rng.randint(0, 4)))
        op = rng.random()
        if op < 0.5:
            t.insert(w); ref[w] += 1
        elif op < 0.7:
            assert t.remove(w) == (ref[w] > 0)
            if ref[w] > 0:
                ref[w] -= 1
        p = w[: rng.randint(0, len(w))]
        assert t.contains(w) == (ref[w] > 0)
        assert t.count_prefix(p) == sum(c for x, c in ref.items() if x.startswith(p))
        k = rng.randint(0, 5)
        assert t.complete(p, k) == sorted(x for x, c in ref.items() if c > 0 and x.startswith(p))[:k]
