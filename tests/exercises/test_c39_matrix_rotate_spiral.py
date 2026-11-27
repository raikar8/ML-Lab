import ast
import inspect
import random

import numpy as np

from helpers import key_of, load

m = load(key_of(__file__))


def test_rotate():
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rows = [id(r) for r in a]
    assert m.rotate(a) is None and a == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
    assert [id(r) for r in a] == rows           # same row objects: in place
    rng = random.Random(39)
    for n in range(0, 7):
        x = [[rng.randint(0, 9) for _ in range(n)] for _ in range(n)]
        exp = np.rot90(np.array(x).reshape(n, n), -1).tolist() if n else []
        m.rotate(x)
        assert x == exp


def ref_spiral(a):
    a = [list(r) for r in a]
    out = []
    while a:
        out += a.pop(0)
        a = [list(r) for r in zip(*a)][::-1]
    return out


def test_spiral():
    assert m.spiral([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
    assert m.spiral([]) == [] and m.spiral([[]]) == [] and m.spiral([[1], [2], [3]]) == [1, 2, 3]
    rng = random.Random(40)
    for _ in range(200):
        R, C = rng.randint(1, 6), rng.randint(1, 6)
        a = [[rng.randint(0, 99) for _ in range(C)] for _ in range(R)]
        assert m.spiral(a) == ref_spiral(a)


def test_set_zeroes():
    rng = random.Random(41)
    for _ in range(300):
        R, C = rng.randint(1, 6), rng.randint(1, 6)
        a = [[rng.choice([0, 1, 2, 3, 4]) for _ in range(C)] for _ in range(R)]
        zr = {i for i in range(R) for j in range(C) if a[i][j] == 0}
        zc = {j for i in range(R) for j in range(C) if a[i][j] == 0}
        exp = [[0 if i in zr or j in zc else a[i][j] for j in range(C)] for i in range(R)]
        m.set_zeroes(a)
        assert a == exp
    src = inspect.getsource(m.set_zeroes)
    assert "set(" not in src
