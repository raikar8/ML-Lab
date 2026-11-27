import ast
import inspect

import numpy as np
import pytest

from helpers import key_of, load

m = load(key_of(__file__))
rng = np.random.default_rng(0)


def test_matmul():
    for _ in range(20):
        a, b, c = rng.integers(1, 7, size=3)
        A, B = rng.normal(size=(a, b)), rng.normal(size=(b, c))
        assert np.allclose(m.matmul(A, B), A @ B)
    with pytest.raises(ValueError):
        m.matmul(np.ones((2, 3)), np.ones((2, 3)))


def test_solve_needs_pivoting():
    A = np.array([[1e-20, 1.0], [1.0, 1.0]])          # naive elimination without pivoting fails here
    b = np.array([1.0, 2.0])
    assert np.allclose(m.solve(A, b), np.linalg.solve(A, b))
    for n in range(1, 8):
        A = rng.normal(size=(n, n)) + n * np.eye(n)
        b = rng.normal(size=n)
        A0, b0 = A.copy(), b.copy()
        assert np.allclose(m.solve(A, b), np.linalg.solve(A, b))
        assert np.array_equal(A, A0) and np.array_equal(b, b0)
    with pytest.raises(ValueError):
        m.solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))


def test_power_iteration():
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    lam = np.array([-7.0, 3.0, 2.0, 1.0, 0.5])
    A = Q @ np.diag(lam) @ Q.T
    val, vec = m.power_iteration(A)
    assert np.isclose(val, -7.0, atol=1e-6)
    ref = Q[:, 0] * np.sign(Q[np.argmax(np.abs(Q[:, 0])), 0])
    assert np.allclose(vec, ref, atol=1e-6) and np.isclose(np.linalg.norm(vec), 1)


def test_project():
    A = rng.normal(size=(10, 3))
    b = rng.normal(size=10)
    p = m.project(A, b)
    assert np.allclose(A.T @ (b - p), 0, atol=1e-9)      # residual orthogonal to the columns
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert np.allclose(p, A @ x)


def test_no_library_shortcuts():
    src = inspect.getsource(m)
    tree = ast.parse(src)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"dot", "matmul", "linalg", "inv", "lstsq"}
    assert not any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult) for n in ast.walk(tree))
