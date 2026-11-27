import pytest
import torch
from helpers import key_of, load

m = load(key_of(__file__))


def tok(t):
    return torch.full((2, 3), float(t)), torch.full((2, 3), -float(t))


def test_append_gather_free():
    c = m.PagedKVCache(4, 4, 2, 3)
    c.new_seq("a")
    for t in range(6):
        c.append("a", *tok(t))
    k, v = c.gather("a")
    assert k.shape == (6, 2, 3) and k[:, 0, 0].tolist() == list(range(6)) and v[5, 1, 2] == -5
    assert c.num_free() == 2
    c.free("a")
    assert c.num_free() == 4


def test_fork_copy_on_write_and_oom():
    c = m.PagedKVCache(4, 4, 2, 3)
    c.new_seq("p")
    for t in range(6):
        c.append("p", *tok(t))
    c.fork("p", "q")
    assert c.num_free() == 2  # shared
    c.append("q", *tok(100))
    c.append("p", *tok(200))
    assert c.gather("q")[0][:, 0, 0].tolist() == [0, 1, 2, 3, 4, 5, 100]
    assert c.gather("p")[0][:, 0, 0].tolist() == [0, 1, 2, 3, 4, 5, 200]
    c.new_seq("r")
    with pytest.raises(MemoryError):
        for t in range(9):
            c.append("r", *tok(t))
