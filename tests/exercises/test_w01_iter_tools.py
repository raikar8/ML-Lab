import itertools

import pytest
from helpers import key_of, load

m = load(key_of(__file__))


def test_batched():
    assert list(m.batched(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(m.batched([], 2)) == []
    assert next(m.batched(itertools.count(), 4)) == [0, 1, 2, 3]  # infinite input
    with pytest.raises(ValueError):
        list(m.batched([1], 0))


def test_sliding_window():
    assert list(m.sliding_window(range(5), 3)) == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert list(m.sliding_window(range(7), 4, step=2)) == [(0, 1, 2, 3), (2, 3, 4, 5)]
    assert list(m.sliding_window([1, 2], 3)) == []


def test_memoize():
    calls = []

    @m.memoize
    def f(x, y=1):
        "doc"
        calls.append((x, y))
        return x + y

    assert f(1) == 2 and f(1) == 2 and f(1, y=2) == 3
    assert calls == [(1, 1), (1, 2)]
    assert f.__name__ == "f" and f.__doc__ == "doc" and len(f.cache) == 2
