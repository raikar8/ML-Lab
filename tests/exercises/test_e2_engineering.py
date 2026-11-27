import dataclasses
import logging

import pytest

from helpers import key_of, load

m = load(key_of(__file__))


def test_valid_and_frozen():
    c = m.TrainConfig.from_dict({"lr": 1, "batch_size": 64, "optimizer": "sgd"})
    assert c == m.TrainConfig(lr=1.0, batch_size=64, optimizer="sgd") and isinstance(c.lr, float)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.lr = 0.5
    assert m.TrainConfig.from_dict({}) == m.TrainConfig()


def test_all_errors_reported_at_once():
    with pytest.raises(m.ConfigError) as e:
        m.TrainConfig.from_dict({"lrr": 0.1, "batch_size": "32", "epochs": 0, "optimizer": "rmsprop",
                                 "seed": True})
    errs = e.value.errors
    assert len(errs) == 5 and isinstance(e.value, ValueError)
    assert any("lrr" in x and "did you mean 'lr'" in x for x in errs)
    assert any("batch_size" in x and "int" in x for x in errs)
    assert any("epochs" in x for x in errs) and any("rmsprop" in x for x in errs) and any("seed" in x for x in errs)
    with pytest.raises(m.ConfigError):
        m.TrainConfig.from_dict({"lr": 0.0})


def test_retry_with_injected_sleep():
    slept, calls = [], []

    @m.retry(times=4, exceptions=(TimeoutError,), base_delay=0.5, sleep=slept.append)
    def flaky():
        """docs"""
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError
        return "ok"

    assert flaky() == "ok" and slept == [0.5, 1.0] and flaky.__name__ == "flaky" and flaky.__doc__ == "docs"

    @m.retry(times=3, exceptions=(TimeoutError,), sleep=slept.append)
    def always():
        raise TimeoutError("nope")

    slept.clear()
    with pytest.raises(TimeoutError, match="nope"):
        always()
    assert len(slept) == 2

    @m.retry(times=5, exceptions=(TimeoutError,), sleep=slept.append)
    def wrong_kind():
        raise KeyError("not retried")

    slept.clear()
    with pytest.raises(KeyError):
        wrong_kind()
    assert slept == []


def test_log_kv(caplog):
    logger = logging.getLogger("e2test")
    with caplog.at_level(logging.INFO, logger="e2test"):
        m.log_kv(logger, logging.INFO, "epoch_end", loss=0.25, epoch=3, note="two words")
    assert caplog.records[-1].getMessage() == "epoch_end epoch=3 loss=0.25 note='two words'"
