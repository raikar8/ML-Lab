"""
title: Engineering hygiene — a validated config, a testable retry decorator, structured logging
week: 0
day: 5
level: medium
time: 40
tags: engineering, testing, config, logging, elective
core: yes

* `TrainConfig` — a **frozen** dataclass with fields `lr: float = 1e-3`, `batch_size: int = 32`, `epochs: int = 10`,
  `optimizer: str = "adamw"`, `seed: int = 0`. `TrainConfig.from_dict(d)` (classmethod) builds one from a dict
  (e.g. parsed YAML), and raises `ConfigError` whose message lists **every** problem at once:
  unknown keys (suggest the closest valid key with `difflib.get_close_matches`), wrong types (ints are accepted for
  float fields; bools are not accepted as numbers), and invalid values (`lr` in (0, 1], `batch_size` ≥ 1, `epochs` ≥ 1,
  `optimizer` in {"sgd", "adam", "adamw"}). `ConfigError` subclasses `ValueError` and has an `errors` list.
* `retry(times, exceptions=(Exception,), base_delay=0.1, sleep=time.sleep)` — a decorator that retries the wrapped
  function up to `times` attempts in total, waiting `base_delay * 2**k` after the k-th failure (exponential backoff,
  using the injected `sleep` so tests don't wait); only the listed exceptions are retried; after the last attempt
  re-raise the last exception. Keep the wrapped function's name and docstring (`functools.wraps`).
* `log_kv(logger, level, event, **fields)` — log one line `event key1=value1 key2=value2` with keys sorted, values
  `repr`'d if they contain spaces; uses the given `logging.Logger`.
"""
import dataclasses
import difflib
import functools
import logging
import time
from dataclasses import dataclass


class ConfigError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("invalid config:\n  - " + "\n  - ".join(self.errors))


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    optimizer: str = "adamw"
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        fields = {f.name: f for f in dataclasses.fields(cls)}
        errors, clean = [], {}
        for key, value in d.items():
            if key not in fields:
                hint = difflib.get_close_matches(key, list(fields), n=1)
                errors.append(f"unknown key '{key}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
                continue
            want = fields[key].type
            want = {"float": float, "int": int, "str": str}.get(want, want)
            ok = (isinstance(value, want) and not isinstance(value, bool)) or \
                 (want is float and isinstance(value, int) and not isinstance(value, bool))
            if not ok:
                errors.append(f"'{key}' must be {want.__name__}, got {type(value).__name__}")
                continue
            clean[key] = float(value) if want is float else value
        checks = {
            "lr": (lambda v: 0 < v <= 1, "must be in (0, 1]"),
            "batch_size": (lambda v: v >= 1, "must be >= 1"),
            "epochs": (lambda v: v >= 1, "must be >= 1"),
            "optimizer": (lambda v: v in {"sgd", "adam", "adamw"}, "must be one of adam, adamw, sgd"),
        }
        for key, (ok, msg) in checks.items():
            if key in clean and not ok(clean[key]):
                errors.append(f"'{key}'={clean[key]!r} {msg}")
        if errors:
            raise ConfigError(errors)
        return cls(**clean)


def retry(times: int, exceptions=(Exception,), base_delay: float = 0.1, sleep=time.sleep):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for k in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if k == times - 1:
                        raise
                    sleep(base_delay * 2 ** k)
        return wrapper
    return deco


def log_kv(logger: logging.Logger, level: int, event: str, **fields) -> None:
    parts = []
    for k in sorted(fields):
        v = fields[k]
        s = str(v)
        parts.append(f"{k}={v!r}" if " " in s else f"{k}={s}")
    logger.log(level, " ".join([event] + parts))
