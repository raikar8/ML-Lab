import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(ms):
    if not ms:
        return (None, 0)
    best = (None, 0)
    for t in sorted({x for s, e in ms for x in (s, e)}):
        c = sum(1 for s, e in ms if s <= t < e)
        if c > best[1]:
            best = (t, c)
    return best


def test_examples():
    assert m.min_rooms([(0, 30), (5, 10), (15, 20)]) == 2
    assert m.busiest([(1, 4), (2, 5), (4, 6)]) == (2, 2)
    assert m.can_attend_all([(0, 10), (10, 20)]) and not m.can_attend_all([(0, 10), (9, 20)])
    assert m.busiest([]) == (None, 0) and m.min_rooms([]) == 0


def test_random_against_brute_force():
    rng = random.Random(28)
    for _ in range(500):
        ms = []
        for _ in range(rng.randint(0, 10)):
            s = rng.randint(0, 20)
            ms.append((s, s + rng.randint(1, 6)))
        assert m.busiest(ms) == brute(ms), ms
        assert m.min_rooms(ms) == brute(ms)[1]
