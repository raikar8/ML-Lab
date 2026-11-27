import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(s):
    # repeatedly delete innermost pairs from the bracket-only string, tracking original indices
    items = [(i, c) for i, c in enumerate(s) if c in "()[]{}"]
    pairs = {"(": ")", "[": "]", "{": "}"}
    changed = True
    while changed:
        changed = False
        for k in range(len(items) - 1):
            if items[k][1] in pairs and pairs[items[k][1]] == items[k + 1][1]:
                del items[k:k + 2]
                changed = True
                break
    if not items:
        return -1
    for idx, c in items:          # first remaining closer is the first error
        if c in ")]}":
            return idx
    return items[0][0]            # only openers left


def test_examples():
    assert m.first_error("a(b[c]{d}e)f") == -1
    assert m.first_error("(]") == 1
    assert m.first_error("([)]") == 2
    assert m.first_error("x((y)") == 1
    assert m.first_error("") == -1 and m.first_error("abc") == -1
    assert m.first_error(")(") == 0
    assert m.is_balanced("{[()()]}") and not m.is_balanced("{[(])}")


def test_random_against_brute_force():
    rng = random.Random(1)
    for _ in range(2000):
        s = "".join(rng.choice("()[]{}ab") for _ in range(rng.randint(0, 14)))
        assert m.first_error(s) == brute(s), s
