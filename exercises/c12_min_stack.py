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
        raise NotImplementedError  # TODO: implement

    def push(self, x):
        raise NotImplementedError  # TODO: implement

    def pop(self):
        raise NotImplementedError  # TODO: implement

    def top(self):
        raise NotImplementedError  # TODO: implement

    def get_min(self):
        raise NotImplementedError  # TODO: implement

    def __len__(self):
        raise NotImplementedError  # TODO: implement


class MaxQueue:
    def __init__(self):
        raise NotImplementedError  # TODO: implement

    def push(self, x):
        raise NotImplementedError  # TODO: implement

    def pop(self):
        raise NotImplementedError  # TODO: implement

    def get_max(self):
        raise NotImplementedError  # TODO: implement

    def __len__(self):
        raise NotImplementedError  # TODO: implement
