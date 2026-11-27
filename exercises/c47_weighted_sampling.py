"""
Random sampling for ML pipelines — weighted pick, Fisher–Yates, weighted sample without replacement

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
        raise NotImplementedError  # TODO: implement

    def pick(self, rng) -> int:
        raise NotImplementedError  # TODO: implement


def shuffle(a: list, rng) -> None:
    raise NotImplementedError  # TODO: implement


def weighted_sample(weights, k: int, rng) -> list[int]:
    raise NotImplementedError  # TODO: implement
