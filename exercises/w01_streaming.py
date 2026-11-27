"""
title: Streaming statistics — Welford, reservoir sampling, top-k frequent
week: 1
day: 4
level: medium
time: 35
tags: python, algorithms, numerics
core: no

1. `RunningStats` — Welford's online algorithm. `update(x)` for one float; properties `n`, `mean`,
   `var` (sample variance, `n-1` denominator; `0.0` when n < 2) and `std`. Must stay accurate for data like
   `1e9 + small noise` where the naive formula fails.
2. `reservoir_sample(stream, k, rng)` — uniform sample of `k` items from an iterable of unknown length using
   `rng.randrange` (a `random.Random`). If the stream has fewer than k items, return them all.
3. `top_k_frequent(items, k)` — the k most frequent items, most frequent first, ties broken by the item value
   (smaller first). Use a heap: O(n log k).
"""

import heapq

import math

from collections import Counter

class RunningStats:
    def __init__(self):
        raise NotImplementedError  # TODO: implement

    def update(self, x: float) -> None:
        raise NotImplementedError  # TODO: implement

    @property
    def mean(self) -> float:
        raise NotImplementedError  # TODO: implement

    @property
    def var(self) -> float:
        raise NotImplementedError  # TODO: implement

    @property
    def std(self) -> float:
        raise NotImplementedError  # TODO: implement


def reservoir_sample(stream, k, rng):
    raise NotImplementedError  # TODO: implement


def top_k_frequent(items, k):
    raise NotImplementedError  # TODO: implement
