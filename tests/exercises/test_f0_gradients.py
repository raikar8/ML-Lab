import numpy as np

from helpers import key_of, load

m = load(key_of(__file__))
rng = np.random.default_rng(0)


def test_numerical_grad_matches_known_gradients():
    A = rng.normal(size=(4, 4)); A = A + A.T
    x = rng.normal(size=4)
    g = m.numerical_grad(lambda v: v @ A @ v, x)
    assert m.rel_error(g, 2 * A @ x) < 1e-7
    W = rng.normal(size=(3, 2))
    x0 = W.copy()
    g = m.numerical_grad(lambda M: np.sum(np.sin(M)), W)
    assert m.rel_error(g, np.cos(W)) < 1e-7 and np.array_equal(W, x0)


def test_logistic_grad_check_and_stability():
    X = rng.normal(size=(50, 3)); y = (rng.random(50) < 0.5).astype(float); w = rng.normal(size=3)
    loss, g = m.logistic_loss_grad(w, X, y)
    num = m.numerical_grad(lambda v: m.logistic_loss_grad(v, X, y)[0], w)
    assert m.rel_error(g, num) < 1e-6
    big_loss, big_g = m.logistic_loss_grad(np.array([1000.0, -1000.0, 0.0]), X, y)
    assert np.isfinite(big_loss) and np.all(np.isfinite(big_g))


def test_gd_and_stable_lr():
    H = np.diag([1.0, 10.0])
    fg = lambda x: (0.5 * x @ H @ x, H @ x)
    lr_max = m.max_stable_lr(H)
    assert np.isclose(lr_max, 0.2)
    x, hist = m.gradient_descent(fg, [1.0, 1.0], 0.9 * lr_max, 200)
    assert len(hist) == 201 and hist[-1] < 1e-6 and hist[0] == 5.5
    _, hist = m.gradient_descent(fg, [1.0, 1.0], 1.1 * lr_max, 50)
    assert hist[-1] > hist[0]                           # diverges above 2/λmax


def test_rel_error_zero_safe():
    assert m.rel_error(np.zeros(3), np.zeros(3)) == 0.0
