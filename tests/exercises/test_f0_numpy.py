import ast
import inspect

import numpy as np

from helpers import key_of, load

m = load(key_of(__file__))
rng = np.random.default_rng(0)


def test_values():
    X, Y = rng.normal(size=(7, 4)), rng.normal(size=(5, 4))
    ref = ((X[:, None, :] - Y[None]) ** 2).sum(-1)
    assert np.allclose(m.pairwise_sq_dists(X, Y), ref) and (m.pairwise_sq_dists(X, X) >= 0).all()
    assert np.array_equal(m.one_hot([2, 0, 1], 3), np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float))
    Z = rng.normal(3, 5, size=(100, 3)); Z[:, 2] = 7.0
    S = m.standardize(Z)
    assert np.allclose(S[:, :2].mean(0), 0) and np.allclose(S[:, :2].std(0), 1) and np.allclose(S[:, 2], 0)
    x = rng.normal(size=50)
    assert np.allclose(m.moving_average(x, 5), np.convolve(x, np.ones(5) / 5, mode="valid"))
    assert np.allclose(m.group_sum([1, 2, 3, 4], [0, 2, 0, 2], 4), [4, 0, 6, 0])
    P = m.softmax_rows(np.array([[1000.0, 1000.0], [0.0, np.log(3)]]))
    assert np.allclose(P, [[0.5, 0.5], [0.25, 0.75]])


def test_no_python_loops():
    tree = ast.parse(inspect.getsource(m))
    loops = (ast.For, ast.While, ast.ListComp, ast.GeneratorExp, ast.DictComp, ast.SetComp)
    assert not any(isinstance(n, loops) for n in ast.walk(tree))


def test_moving_average_is_linear_time():
    import time
    x = rng.normal(size=2_000_000)
    t0 = time.perf_counter()
    m.moving_average(x, 50_000)
    assert time.perf_counter() - t0 < 1.0
