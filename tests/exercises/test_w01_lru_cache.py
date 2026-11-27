import inspect

from helpers import key_of, load

m = load(key_of(__file__))


def test_basic_eviction():
    c = m.LRUCache(2)
    c.put(1, 1); c.put(2, 2)
    assert c.get(1) == 1          # 1 is now most recent
    c.put(3, 3)                   # evicts 2
    assert c.get(2) == -1 and c.get(3) == 3 and len(c) == 2
    c.put(1, 10)                  # update
    assert c.keys() == [1, 3] and c.get(1) == 10


def test_capacity_zero_and_no_ordereddict():
    c = m.LRUCache(0)
    c.put(1, 1)
    assert c.get(1) == -1
    assert "OrderedDict(" not in inspect.getsource(m) and not hasattr(m, "OrderedDict")


def test_many_ops():
    c = m.LRUCache(100)
    for i in range(1000):
        c.put(i, i)
    assert len(c) == 100 and c.keys()[0] == 999 and c.get(899) == -1 and c.get(900) == 900
