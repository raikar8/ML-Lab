import pytest

from helpers import key_of, load

m = load(key_of(__file__))


def make_cfg_class():
    class Cfg:
        lr = m.Bounded(0.0, 1.0)
        layers = m.Bounded(1, 64)

        def __init__(self, lr, layers):
            self.lr, self.layers = lr, layers
    return Cfg


def test_bounded():
    Cfg = make_cfg_class()
    c = Cfg(0.1, 4)
    assert c.lr == 0.1 and c.__dict__ == {"lr": 0.1, "layers": 4}
    assert isinstance(Cfg.lr, m.Bounded)
    with pytest.raises(ValueError):
        c.lr = 1.5
    with pytest.raises(TypeError):
        c.layers = "4"
    with pytest.raises(TypeError):
        c.layers = True
    class Empty:
        v = m.Bounded(0, 1)
    with pytest.raises(AttributeError):
        Empty().v
    c.__dict__["lr"] = 99            # a data descriptor still wins on assignment
    with pytest.raises(ValueError):
        c.lr = 99


def test_lazy():
    calls = []

    class Model:
        @m.lazy
        def weights(self):
            """expensive"""
            calls.append(1)
            return [1, 2, 3]

    a = Model()
    assert a.weights == [1, 2, 3] and a.weights is a.weights and len(calls) == 1
    assert "weights" in a.__dict__ and Model.weights.__doc__ == "expensive"
    del a.weights
    a.weights
    assert len(calls) == 2
    Model().weights
    assert len(calls) == 3


def test_vec2():
    v, w = m.Vec2(1, 2), m.Vec2(3, 4)
    assert repr(v) == "Vec2(1.0, 2.0)" and v + w == m.Vec2(4, 6) and w - v == m.Vec2(2, 2)
    assert 2 * v == v * 2 == m.Vec2(2, 4) and abs(w) == 5.0
    x, y = w
    assert (x, y) == (3.0, 4.0)
    assert {v, m.Vec2(1.0, 2.0)} == {v} and v != (1, 2)
    with pytest.raises(AttributeError):
        v.x = 5
    with pytest.raises(AttributeError):
        v.z = 1
    with pytest.raises(TypeError):
        v + 1
    with pytest.raises(TypeError):
        v * "a"
    assert not hasattr(v, "__dict__")
