"""
title: SPSC ring buffer with overrun/underrun accounting for real-time audio
week: 20
day: 5
level: medium
time: 30
tags: python, real-time, audio, embedded
core: yes

`RingBuffer(capacity)` — capacity must be a power of two (else `ValueError`); stores floats in a preallocated NumPy array
and uses monotonically increasing `write_idx`/`read_idx` with masking (`idx & (capacity−1)`).

* `write(samples)` → number written; if there's not enough space, write what fits and add the dropped count to
  `overruns`.
* `read(n)` → NumPy array of exactly `n` samples; if fewer are available, return what's there padded with zeros and add the
  missing count to `underruns`.
* `available()` → samples ready to read; `free()` → free space.
Wrap-around must work for arbitrarily many writes/reads.
"""

import numpy as np

class RingBuffer:
    def __init__(self, capacity):
        raise NotImplementedError  # TODO: implement

    def available(self):
        raise NotImplementedError  # TODO: implement

    def free(self):
        raise NotImplementedError  # TODO: implement

    def write(self, samples):
        raise NotImplementedError  # TODO: implement

    def read(self, n):
        raise NotImplementedError  # TODO: implement
