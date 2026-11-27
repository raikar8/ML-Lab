"""
title: Rate limiters — token bucket and sliding-window log
pattern: Design
level: medium
time: 30
frequency: high
week: 20
day: 4
tags: design, queue, time, systems
complexity: token bucket O(1) per request; sliding log O(1) amortised per request, O(limit) memory per key
hint: Token bucket: tokens refill continuously at `rate` per second up to `capacity`; a request spends one token.
hint: Sliding log: keep a deque of accepted timestamps per key; drop those older than the window, accept if fewer than the limit remain.
follow-up: Distributed rate limiting across many servers (central store with atomic ops, or per-node budgets).
follow-up: Rate-limit an on-device wake-word detector's false triggers — which algorithm and why?

Time is passed in explicitly (seconds, float, non-decreasing per key) so tests are deterministic.

* `TokenBucket(rate, capacity)` — starts full. `allow(now) -> bool`: refill `rate × elapsed` tokens (capped at
  `capacity`), then spend 1 token if at least 1 is available.
* `SlidingWindowLimiter(limit, window)` — `allow(key, now) -> bool`: accept if fewer than `limit` requests from
  `key` were **accepted** in the half-open interval `(now − window, now]`. Rejected requests don't count.
"""
from collections import deque


class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate, self.capacity = rate, capacity
        self.tokens = float(capacity)
        self.last = None

    def allow(self, now: float) -> bool:
        if self.last is not None:
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1.0 - 1e-12:
            self.tokens -= 1.0
            return True
        return False


class SlidingWindowLimiter:
    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self.logs: dict = {}

    def allow(self, key, now: float) -> bool:
        q = self.logs.setdefault(key, deque())
        while q and q[0] <= now - self.window:
            q.popleft()
        if len(q) < self.limit:
            q.append(now)
            return True
        return False
