import random
import time
import types

from helpers import key_of, load

m = load(key_of(__file__))


def test_dedupe():
    assert m.dedupe([3, 1, 3, 2, 1]) == [3, 1, 2] and m.dedupe([]) == []
    big = list(range(100_000)) * 2
    t0 = time.perf_counter()
    assert m.dedupe(big) == list(range(100_000))
    assert time.perf_counter() - t0 < 1


def test_flatten():
    g = m.flatten([1, [2, [3, (4, 5)], "ab"], [], [[6]]])
    assert isinstance(g, types.GeneratorType)
    assert list(g) == [1, 2, 3, (4, 5), "ab", 6]
    deep = x = []
    for i in range(10_000):
        nxt = [i]
        x.append(nxt)
        x = nxt
    assert len(list(m.flatten(deep))) == 10_000


def test_group_and_rle():
    assert m.group_by(["apple", "bob", "avocado", "cat"], lambda w: w[0]) == {"a": ["apple", "avocado"], "b": ["bob"], "c": ["cat"]}
    assert m.rle_encode("aaabcc") == [("a", 3), ("b", 1), ("c", 2)] and m.rle_encode("") == []
    rng = random.Random(0)
    for _ in range(200):
        s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 20)))
        assert m.rle_decode(m.rle_encode(s)) == s


def test_rank_and_grid():
    rows = [("ann", 10, 5.0), ("bob", 12, 9.0), ("cy", 10, 4.0), ("al", 10, 4.0)]
    assert m.rank_players(rows) == ["bob", "al", "cy", "ann"]
    g = m.make_grid(3, 2)
    g[0][0] = 1
    assert g == [[1, 0], [0, 0], [0, 0]]
