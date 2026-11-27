"""
Rate limiters — token bucket and sliding-window log

Time is passed in explicitly (seconds, float, non-decreasing per key) so tests are deterministic.

* `TokenBucket(rate, capacity)` — starts full. `allow(now) -> bool`: refill `rate × elapsed` tokens (capped at
  `capacity`), then spend 1 token if at least 1 is available.
* `SlidingWindowLimiter(limit, window)` — `allow(key, now) -> bool`: accept if fewer than `limit` requests from
  `key` were **accepted** in the half-open interval `(now − window, now]`. Rejected requests don't count.
"""

from collections import deque

class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        raise NotImplementedError  # TODO: implement

    def allow(self, now: float) -> bool:
        raise NotImplementedError  # TODO: implement


class SlidingWindowLimiter:
    def __init__(self, limit: int, window: float):
        raise NotImplementedError  # TODO: implement

    def allow(self, key, now: float) -> bool:
        raise NotImplementedError  # TODO: implement
