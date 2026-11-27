from helpers import key_of, load

m = load(key_of(__file__))

T = [("in", 1000, 0, 1), ("a", 4000, 1, 2), ("b", 4000, 2, 3), ("c", 1000, 3, 4), ("out", 100, 4, 5)]


def test_peak_and_plan():
    assert m.peak_live_bytes(T) == 8000
    offs, size = m.plan_arena(T)
    assert size <= 8000 + 64 and size >= 8000
    for i, (n1, s1, a1, b1) in enumerate(T):
        assert offs[n1] % 16 == 0
        for n2, s2, a2, b2 in T[i + 1:]:
            if not (b1 < a2 or b2 < a1):
                assert offs[n1] + s1 <= offs[n2] or offs[n2] + s2 <= offs[n1], (n1, n2)
    assert size < sum(t[1] for t in T)
