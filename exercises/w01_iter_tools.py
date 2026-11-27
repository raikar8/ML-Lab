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
    raise NotImplementedError  # TODO: implement


def sliding_window(seq, size, step=1):
    raise NotImplementedError  # TODO: implement


def memoize(fn):
    raise NotImplementedError  # TODO: implement
