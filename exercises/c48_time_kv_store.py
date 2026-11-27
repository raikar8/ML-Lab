"""
title: Time-versioned key–value store (get the value as of a timestamp)
pattern: Design
level: medium
time: 25
frequency: high
week: 24
day: 4
tags: design, binary-search, versioning
complexity: set O(1) amortised (timestamps increase per key), get O(log m) for m versions of a key
hint: Per key, keep parallel lists of timestamps and values; timestamps arrive in increasing order, so they stay sorted.
hint: get(key, t) = the value at the last timestamp ≤ t → bisect_right − 1.
follow-up: Out-of-order writes — what structure then? (balanced tree / sorted list with insort, or an LSM-style log)
follow-up: This is how a feature store serves point-in-time-correct training data (no label leakage). Explain.

`TimeMap` with

* `set(key, value, timestamp)` — timestamps for a given key are strictly increasing across calls;
* `get(key, timestamp)` — the value with the largest timestamp `<= timestamp`, or `None`;
* `history(key, t0, t1)` — list of `(timestamp, value)` with `t0 <= timestamp <= t1`, in order.

`get` and `history` must use binary search (no linear scan over versions).
"""

import bisect

class TimeMap:
    def __init__(self):
        raise NotImplementedError  # TODO: implement

    def set(self, key, value, timestamp) -> None:
        raise NotImplementedError  # TODO: implement

    def get(self, key, timestamp):
        raise NotImplementedError  # TODO: implement

    def history(self, key, t0, t1):
        raise NotImplementedError  # TODO: implement
