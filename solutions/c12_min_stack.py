"""
title: Min-stack and max-queue — O(1) extremes
pattern: Design
level: easy
time: 25
frequency: high
week: 6
day: 4
tags: stack, deque, monotonic, design
complexity: O(1) per operation (amortised O(1) for the queue)
hint: MinStack: push (value, min so far) pairs — the min below never changes while an item sits on top.
hint: MaxQueue: keep a deque of candidates in decreasing order; pop smaller ones from the back on push.
follow-up: Implement the queue with two stacks, each carrying its own running max.
follow-up: Which audio-processing primitive is MaxQueue? (running max over a sliding window → peak meters, limiters)

* `MinStack`: `push(x)`, `pop() -> x`, `top() -> x`, `get_min() -> x`, `__len__`, all O(1).
* `MaxQueue`: FIFO with `push(x)`, `pop() -> x` (front), `get_max() -> x`, `__len__`, amortised O(1).

`pop`/`top`/`get_min`/`get_max` on an empty structure raise `IndexError`.
"""
from collections import deque


class MinStack:
    def __init__(self):
        self._items: list[tuple[int, int]] = []  # (value, min of stack up to here)

    def push(self, x):
        m = x if not self._items else min(x, self._items[-1][1])
        self._items.append((x, m))

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty MinStack")
        return self._items.pop()[0]

    def top(self):
        if not self._items:
            raise IndexError("top of empty MinStack")
        return self._items[-1][0]

    def get_min(self):
        if not self._items:
            raise IndexError("min of empty MinStack")
        return self._items[-1][1]

    def __len__(self):
        return len(self._items)


class MaxQueue:
    def __init__(self):
        self._q: deque = deque()
        self._cand: deque = deque()  # non-increasing candidates for the max

    def push(self, x):
        self._q.append(x)
        while self._cand and self._cand[-1] < x:
            self._cand.pop()
        self._cand.append(x)

    def pop(self):
        if not self._q:
            raise IndexError("pop from empty MaxQueue")
        x = self._q.popleft()
        if x == self._cand[0]:
            self._cand.popleft()
        return x

    def get_max(self):
        if not self._q:
            raise IndexError("max of empty MaxQueue")
        return self._cand[0]

    def __len__(self):
        return len(self._q)
