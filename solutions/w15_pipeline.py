"""
title: Pipeline schedules — GPipe vs 1F1B, bubble fraction and peak activations
week: 15
day: 4
level: medium
time: 35
tags: python, distributed, pipeline
core: no

Simulate `p` stages and `m` micro-batches where a forward and a backward each take 1 time unit.

* `gpipe_schedule(p, m)` / `one_f_one_b_schedule(p, m)` → list (per stage) of lists of `(time, "F"|"B", mb)`.
  Rules: F of micro-batch `j` on stage `s` needs F of `j` on `s−1` done; B of `j` on stage `s` needs B of `j` on `s+1`
  done (or F of `j` on the last stage). GPipe: all forwards then all backwards. 1F1B: stage `s` does `p−s` warm-up
  forwards (capped at m), then alternates one backward and one forward, then drains backwards. Each stage runs ops
  as soon as allowed, in its order.
* `bubble_fraction(schedule)` → idle time / (stages × makespan). `peak_in_flight(schedule)` → the max number of
  micro-batches whose forward is done but backward not yet done, on any stage.
"""


def simulate(orders):
    p = len(orders)
    done = {}
    t_free = [0] * p
    sched = [[] for _ in range(p)]
    ptr = [0] * p
    remaining = sum(len(o) for o in orders)
    while remaining:
        progressed = False
        for s in range(p):
            if ptr[s] >= len(orders[s]):
                continue
            kind, j = orders[s][ptr[s]]
            if kind == "F":
                dep = done.get(("F", s - 1, j)) if s > 0 else 0
            else:
                dep = done.get(("B", s + 1, j)) if s < p - 1 else done.get(("F", s, j))
            if dep is None:
                continue
            start = max(t_free[s], dep)
            done[(kind, s, j)] = start + 1
            t_free[s] = start + 1
            sched[s].append((start, kind, j))
            ptr[s] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("deadlock")
    return sched


def gpipe_schedule(p, m):
    return simulate([[("F", j) for j in range(m)] + [("B", j) for j in range(m)] for _ in range(p)])


def one_f_one_b_schedule(p, m):
    orders = []
    for s in range(p):
        warm = min(p - s, m)
        o = [("F", j) for j in range(warm)]
        f, b = warm, 0
        while b < m:
            o.append(("B", b)); b += 1
            if f < m:
                o.append(("F", f)); f += 1
        orders.append(o)
    return simulate(orders)


def bubble_fraction(schedule):
    makespan = max(t + 1 for st in schedule for t, _, _ in st)
    busy = sum(len(st) for st in schedule)
    return 1 - busy / (len(schedule) * makespan)


def peak_in_flight(schedule):
    peak = 0
    for st in schedule:
        live = 0
        for _, kind, _ in sorted(st):
            live += 1 if kind == "F" else -1
            peak = max(peak, live)
    return peak
