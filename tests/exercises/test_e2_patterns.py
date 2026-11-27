import numpy as np
import pytest

from helpers import key_of, load

m = load(key_of(__file__))


def test_registry():
    r = m.Registry("model")

    @r.register("tiny")
    class Tiny:
        def __init__(self, width=8):
            self.width = width

    assert r.build("tiny", width=4).width == 4 and "tiny" in r and r.names() == ["tiny"]
    assert Tiny.__name__ == "Tiny"
    with pytest.raises(KeyError):
        r.register("tiny")(object)
    with pytest.raises(KeyError, match="tiny"):
        r.build("huge")


def test_loss_strategies():
    losses = m.default_losses()
    assert losses.names() == ["huber", "mse"]
    p, t = np.array([0.0, 2.0, 5.0]), np.array([0.0, 0.0, 0.0])
    assert losses.build("mse")(p, t) == pytest.approx((0 + 4 + 25) / 3)
    assert losses.build("huber", delta=1.0)(p, t) == pytest.approx((0 + 1.5 + 4.5) / 3)
    for name in losses.names():                 # interchangeable strategies
        assert losses.build(name)(t, t) == 0


def test_trainer_with_callbacks():
    losses = [1.0, 0.8, 0.79, 0.85, 0.80, 0.81, 0.5]
    es, hist = m.EarlyStopping(patience=3, min_delta=0.02), m.History()
    tr = m.Trainer(lambda e: losses[e], [hist, es])
    out = tr.fit(len(losses))
    assert out == [1.0, 0.8, 0.79, 0.85, 0.80]    # 0.79 is not a 0.02 improvement over 0.8
    assert es.best == 0.8 and es.best_epoch == 1 and tr.stop
    assert hist.records == list(enumerate(out))
    tr2 = m.Trainer(lambda e: 1.0 / (e + 1))
    assert len(tr2.fit(5)) == 5 and not tr2.stop
