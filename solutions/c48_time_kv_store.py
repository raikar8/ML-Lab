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
        self.times: dict = {}
        self.values: dict = {}

    def set(self, key, value, timestamp) -> None:
        self.times.setdefault(key, []).append(timestamp)
        self.values.setdefault(key, []).append(value)

    def get(self, key, timestamp):
        ts = self.times.get(key)
        if not ts:
            return None
        i = bisect.bisect_right(ts, timestamp) - 1
        return self.values[key][i] if i >= 0 else None

    def history(self, key, t0, t1):
        ts = self.times.get(key)
        if not ts:
            return []
        lo, hi = bisect.bisect_left(ts, t0), bisect.bisect_right(ts, t1)
        return list(zip(ts[lo:hi], self.values[key][lo:hi]))
