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
        self.n, self._mean, self._m2 = 0, 0.0, 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self._mean
        self._mean += d / self.n
        self._m2 += d * (x - self._mean)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def var(self) -> float:
        return self._m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)


def reservoir_sample(stream, k, rng):
    res = []
    for i, x in enumerate(stream):
        if i < k:
            res.append(x)
        else:
            j = rng.randrange(i + 1)
            if j < k:
                res[j] = x
    return res


def top_k_frequent(items, k):
    cnt = Counter(items)
    best = heapq.nsmallest(k, cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return [x for x, _ in best]
