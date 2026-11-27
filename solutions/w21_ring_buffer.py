"""
title: SPSC ring buffer with overrun/underrun accounting for real-time audio
week: 21
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
        if capacity <= 0 or capacity & (capacity - 1):
            raise ValueError("capacity must be a power of two")
        self.cap, self.mask = capacity, capacity - 1
        self.buf = np.zeros(capacity)
        self.write_idx = self.read_idx = 0
        self.overruns = self.underruns = 0

    def available(self):
        return self.write_idx - self.read_idx

    def free(self):
        return self.cap - self.available()

    def write(self, samples):
        samples = np.asarray(samples, dtype=float)
        n = min(len(samples), self.free())
        self.overruns += len(samples) - n
        for i in range(n):
            self.buf[(self.write_idx + i) & self.mask] = samples[i]
        self.write_idx += n
        return n

    def read(self, n):
        k = min(n, self.available())
        out = np.zeros(n)
        for i in range(k):
            out[i] = self.buf[(self.read_idx + i) & self.mask]
        self.read_idx += k
        self.underruns += n - k
        return out
