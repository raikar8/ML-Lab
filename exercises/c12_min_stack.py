"""
Min-stack and max-queue — O(1) extremes

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
