"""
title: Concurrency in practice — thread/process maps, a thread-safe counter, an ordered worker pipeline
week: 0
day: 1
level: medium
time: 45
tags: python, concurrency, gil, threads, elective
core: yes

* `parallel_map(fn, items, workers=4, kind="thread")` — apply `fn` to every item with a
  `ThreadPoolExecutor` (`kind="thread"`) or `ProcessPoolExecutor` (`kind="process"`); return results **in input
  order**; re-raise the first exception (in input order) from `fn`. Raise `ValueError` for another `kind`.
* `Counter` — `increment(n=1)` and `value` (property), safe to call from many threads at once (use a `Lock`).
* `ordered_pipeline(items, fn, workers=4, maxsize=8)` — classic producer/consumer: one producer thread puts
  `(index, item)` on a **bounded** `queue.Queue(maxsize)`; `workers` threads take items, compute `fn(item)` and store the
  result; shut down with one `None` sentinel per worker. Return results in input order. If `fn` raises, the pipeline
  must still shut down (no hung threads) and re-raise that exception in the caller.
* `speedup(serial_s, parallel_s)` — `serial_s / parallel_s`.

The tests check ordering, exceptions, that I/O-like work (sleeps) overlaps with threads, and that the counter
never loses updates.
"""

import queue

import threading

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def parallel_map(fn, items, workers: int = 4, kind: str = "thread"):
    raise NotImplementedError  # TODO: implement


class Counter:
    def __init__(self):
        raise NotImplementedError  # TODO: implement

    def increment(self, n: int = 1) -> None:
        raise NotImplementedError  # TODO: implement

    @property
    def value(self) -> int:
        raise NotImplementedError  # TODO: implement


def ordered_pipeline(items, fn, workers: int = 4, maxsize: int = 8):
    raise NotImplementedError  # TODO: implement


def speedup(serial_s: float, parallel_s: float) -> float:
    raise NotImplementedError  # TODO: implement
