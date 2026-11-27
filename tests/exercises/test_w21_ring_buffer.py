import numpy as np
import pytest
from helpers import key_of, load

m = load(key_of(__file__))


def test_ring_buffer():
    with pytest.raises(ValueError):
        m.RingBuffer(100)
    rb = m.RingBuffer(8)
    assert rb.write(np.arange(5)) == 5 and rb.available() == 5
    np.testing.assert_array_equal(rb.read(3), [0, 1, 2])
    assert rb.write(np.arange(10, 20)) == 6 and rb.overruns == 4
    np.testing.assert_array_equal(rb.read(10), [3, 4, 10, 11, 12, 13, 14, 15, 0, 0])
    assert rb.underruns == 2
    total = []
    for k in range(200):
        rb.write(np.arange(k * 3, k * 3 + 3)); total += rb.read(3).tolist()
    assert total == list(range(600))
