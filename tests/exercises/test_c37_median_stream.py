import random
import statistics
import time

import pytest

from helpers import key_of, load

m = load(key_of(__file__))


def test_examples():
    mf = m.MedianFinder()
    with pytest.raises(ValueError):
        mf.median()
    mf.add(1); mf.add(2)
    assert mf.median() == 1.5
    mf.add(3)
    assert mf.median() == 2.0 and len(mf) == 3


def test_random_against_statistics():
    rng = random.Random(37)
    mf, seen = m.MedianFinder(), []
    for _ in range(3000):
        x = rng.randint(-100, 100)
        mf.add(x); seen.append(x)
        assert mf.median() == statistics.median(seen)


def test_log_time():
    mf = m.MedianFinder()
    t0 = time.perf_counter()
    for i in range(200_000):
        mf.add((i * 7919) % 100_003)
        if i % 1000 == 0:
            mf.median()
    assert time.perf_counter() - t0 < 3
