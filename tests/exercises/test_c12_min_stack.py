import random
from collections import deque

import pytest

from helpers import key_of, load

m = load(key_of(__file__))


def test_min_stack():
    s = m.MinStack()
    for x in [5, 3, 7, 3, 1]:
        s.push(x)
    assert s.get_min() == 1 and s.pop() == 1 and s.get_min() == 3
    assert s.pop() == 3 and s.get_min() == 3 and s.top() == 7 and len(s) == 3
    with pytest.raises(IndexError):
        m.MinStack().pop()


def test_max_queue():
    q = m.MaxQueue()
    for x in [2, 5, 1, 5, 3]:
        q.push(x)
    assert q.get_max() == 5 and q.pop() == 2 and q.get_max() == 5
    assert q.pop() == 5 and q.get_max() == 5 and q.pop() == 1 and q.pop() == 5 and q.get_max() == 3
    with pytest.raises(IndexError):
        m.MaxQueue().get_max()


def test_random_ops():
    rng = random.Random(12)
    s, ref = m.MinStack(), []
    q, refq = m.MaxQueue(), deque()
    for _ in range(5000):
        if rng.random() < 0.6 or not ref:
            x = rng.randint(-50, 50); s.push(x); ref.append(x)
        else:
            assert s.pop() == ref.pop()
        if ref:
            assert s.get_min() == min(ref) and s.top() == ref[-1]
        if rng.random() < 0.6 or not refq:
            x = rng.randint(-50, 50); q.push(x); refq.append(x)
        else:
            assert q.pop() == refq.popleft()
        if refq:
            assert q.get_max() == max(refq)
        assert len(q) == len(refq)
