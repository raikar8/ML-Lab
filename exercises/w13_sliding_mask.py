"""
title: Sliding-window + attention-sink masks and a rolling KV cache
week: 13
day: 1
level: medium
time: 30
tags: pytorch, attention, long-context
core: no

* `sliding_window_mask(T, window, n_sink=0)` → bool `(T, T)`, True = may attend: query `i` sees keys `j` with
  `i − window < j ≤ i`, plus the first `n_sink` positions (if `j ≤ i`).
* `RollingKVCache(window, n_sink)` — keeps the first `n_sink` keys/values forever and at most `window` most recent ones.
  `append(k, v)` with `k, v (B, H, 1, D)`; `get()` → `(k, v)` concatenated `[sinks, recent]`; `positions()` → list of
  absolute token positions currently held. This is StreamingLLM's cache.
"""

import torch

def sliding_window_mask(T, window, n_sink=0):
    raise NotImplementedError  # TODO: implement


class RollingKVCache:
    def __init__(self, window, n_sink):
        raise NotImplementedError  # TODO: implement

    def append(self, k, v):
        raise NotImplementedError  # TODO: implement

    def get(self):
        raise NotImplementedError  # TODO: implement

    def positions(self):
        raise NotImplementedError  # TODO: implement
