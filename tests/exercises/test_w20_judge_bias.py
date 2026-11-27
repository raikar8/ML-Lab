from helpers import key_of, load

m = load(key_of(__file__))


def test_judge():
    ab = ["first", "first", "second", "tie"]
    ba = ["first", "second", "first", "tie"]
    assert m.position_bias(ab, ba) == 0.25
    assert m.debiased_win_rate(ab, ba) == (1 + 1 + 0 + 0.5 + 0 + 1 + 0 + 0.5) / 8
    assert m.cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert abs(m.cohens_kappa([1, 0, 1, 0], [1, 1, 0, 0])) < 1e-12
