import math
import threading
import time

import pytest

from helpers import key_of, load

m = load(key_of(__file__))


def slow_square(x):
    time.sleep(0.02)
    return x * x


def test_parallel_map_order_and_overlap():
    t0 = time.perf_counter()
    out = m.parallel_map(slow_square, range(40), workers=10)
    dt = time.perf_counter() - t0
    assert out == [x * x for x in range(40)]
    assert dt < 0.4                                # 40 × 20 ms serial = 0.8 s; sleeps release the GIL


def test_parallel_map_process_and_errors():
    assert m.parallel_map(math.factorial, [5, 6, 7], workers=2, kind="process") == [120, 720, 5040]

    def boom(x):
        if x in (3, 5):
            raise KeyError(x)
        return x

    with pytest.raises(KeyError) as e:
        m.parallel_map(boom, range(8))
    assert e.value.args == (3,)
    with pytest.raises(ValueError):
        m.parallel_map(abs, [1], kind="gpu")


def test_counter_is_thread_safe():
    c = m.Counter()

    def hammer():
        for _ in range(20_000):
            c.increment()

    ts = [threading.Thread(target=hammer) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert c.value == 160_000


def test_ordered_pipeline():
    out = m.ordered_pipeline(range(50), slow_square, workers=8, maxsize=4)
    assert out == [x * x for x in range(50)]
    assert m.ordered_pipeline([], abs) == []


def test_pipeline_error_shuts_down():
    def bad(x):
        if x == 7:
            raise RuntimeError("bad item")
        time.sleep(0.001)
        return x

    before = threading.active_count()
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="bad item"):
        m.ordered_pipeline(range(1000), bad, workers=4, maxsize=2)
    assert time.perf_counter() - t0 < 5
    time.sleep(0.1)
    assert threading.active_count() <= before      # no leaked threads


def test_speedup():
    assert m.speedup(2.0, 0.5) == 4.0
