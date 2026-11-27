from helpers import key_of, load

m = load(key_of(__file__))


def test_schedules():
    p, mb = 4, 8
    g, o = m.gpipe_schedule(p, mb), m.one_f_one_b_schedule(p, mb)
    for sch in (g, o):
        assert all(len(st) == 2 * mb for st in sch)
    assert abs(m.bubble_fraction(g) - (p - 1) / (mb + p - 1)) < 1e-9
    assert m.peak_in_flight(g) == mb and m.peak_in_flight(o) == p
    assert m.bubble_fraction(o) <= m.bubble_fraction(g) + 1e-9
