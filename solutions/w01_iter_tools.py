"""
title: Iterator tools — batched, sliding_window, memoize
week: 1
day: 2
level: easy
time: 25
tags: python, generators, decorators
core: no

Implement three small utilities you'll reuse in data pipelines.

1. `batched(iterable, n)` — a **generator** yielding lists of up to `n` items; the last batch may be shorter.
   Must work on any iterable (including infinite generators — don't call `len`). `n < 1` → `ValueError`.
2. `sliding_window(seq, size, step=1)` — generator of tuples of length `size`, starting every `step` items
   (like framing audio). Windows that would run past the end are dropped.
3. `memoize` — a decorator caching results by positional + keyword arguments (hashable), exposing
   `wrapper.cache` (a dict) and preserving `__name__`/`__doc__` via `functools.wraps`.
"""
import functools
from collections import deque
from itertools import islice


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def sliding_window(seq, size, step=1):
    it = iter(seq)
    win = deque(islice(it, size), maxlen=size)
    if len(win) < size:
        return
    yield tuple(win)
    while True:
        new = list(islice(it, step))
        if len(new) < step:
            return
        win.extend(new)
        yield tuple(win)


def memoize(fn):
    cache = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache
    return wrapper
