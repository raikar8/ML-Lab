"""
title: Random sampling for ML pipelines — weighted pick, Fisher–Yates, weighted sample without replacement
pattern: Randomised
level: medium
time: 30
frequency: high
week: 24
day: 2
tags: randomised, prefix-sums, binary-search, sampling
complexity: weighted pick O(n) build + O(log n) per draw; shuffle O(n); A-ES sampling O(n log k)
hint: Weighted pick: cumulative sums, draw u ∈ [0, total), binary-search the first prefix > u.
hint: Fisher–Yates: for i from n−1 down to 1, swap a[i] with a[j], j uniform in [0, i]. Weighted without replacement: key = u^(1/w), keep the k largest.
follow-up: Alias method: O(1) per draw after O(n) setup — when is it worth it? (millions of draws, fixed weights)
follow-up: Why is "sort by random()" or swapping with j ∈ [0, n) a biased shuffle?

All randomness comes from a `random.Random` instance `rng` passed in (so results are testable).

* `WeightedPicker(weights)` — non-negative weights, positive total; `pick(rng)` returns index `i` with probability
  `w[i] / sum(w)` using **one** `rng.random()` call and a binary search (bisect allowed).
* `shuffle(a, rng)` — in-place Fisher–Yates using `rng.randrange`.
* `weighted_sample(weights, k, rng)` — `k` distinct indices sampled without replacement where each next pick is
  proportional to the remaining weights (Efraimidis–Spirakis: key `u ** (1/w)`, take the top k). Zero-weight
  items are never chosen; raise `ValueError` if fewer than `k` items have positive weight.
"""
import bisect
import heapq
import itertools


class WeightedPicker:
    def __init__(self, weights):
        self.cum = list(itertools.accumulate(weights))
        self.total = self.cum[-1]
        if self.total <= 0:
            raise ValueError("total weight must be positive")

    def pick(self, rng) -> int:
        u = rng.random() * self.total
        return bisect.bisect_right(self.cum, u)


def shuffle(a: list, rng) -> None:
    for i in range(len(a) - 1, 0, -1):
        j = rng.randrange(i + 1)
        a[i], a[j] = a[j], a[i]


def weighted_sample(weights, k: int, rng) -> list[int]:
    if sum(1 for w in weights if w > 0) < k:
        raise ValueError("not enough items with positive weight")
    keyed = ((rng.random() ** (1.0 / w), i) for i, w in enumerate(weights) if w > 0)
    return [i for _, i in heapq.nlargest(k, keyed)]
