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
    raise NotImplementedError  # TODO: implement


def gpipe_schedule(p, m):
    raise NotImplementedError  # TODO: implement


def one_f_one_b_schedule(p, m):
    raise NotImplementedError  # TODO: implement


def bubble_fraction(schedule):
    raise NotImplementedError  # TODO: implement


def peak_in_flight(schedule):
    raise NotImplementedError  # TODO: implement
