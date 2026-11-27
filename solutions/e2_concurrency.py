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
    if kind == "thread":
        pool = ThreadPoolExecutor(max_workers=workers)
    elif kind == "process":
        pool = ProcessPoolExecutor(max_workers=workers)
    else:
        raise ValueError("kind must be 'thread' or 'process'")
    with pool:
        futures = [pool.submit(fn, x) for x in items]
        return [f.result() for f in futures]       # .result() re-raises, in input order


class Counter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


def ordered_pipeline(items, fn, workers: int = 4, maxsize: int = 8):
    items = list(items)
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    results = [None] * len(items)
    errors: list[tuple[int, BaseException]] = []
    err_lock = threading.Lock()
    stop = threading.Event()

    def producer():
        for pair in enumerate(items):
            while not stop.is_set():
                try:
                    q.put(pair, timeout=0.05)
                    break
                except queue.Full:
                    continue
            if stop.is_set():
                break
        for _ in range(workers):
            q.put(None)

    def worker():
        while True:
            job = q.get()
            if job is None:
                return
            i, x = job
            if stop.is_set():
                continue                           # drain without working after a failure
            try:
                results[i] = fn(x)
            except BaseException as e:             # noqa: BLE001 — re-raised in the caller
                with err_lock:
                    errors.append((i, e))
                stop.set()

    threads = [threading.Thread(target=producer)] + [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise min(errors, key=lambda t: t[0])[1]
    return results


def speedup(serial_s: float, parallel_s: float) -> float:
    return serial_s / parallel_s
