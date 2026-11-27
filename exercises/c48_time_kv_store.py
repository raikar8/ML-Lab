"""
Time-versioned key–value store (get the value as of a timestamp)

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
