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
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    causal = j <= i
    return causal & ((j > i - window) | (j < n_sink))


class RollingKVCache:
    def __init__(self, window, n_sink):
        self.window, self.n_sink = window, n_sink
        self.sk = self.sv = self.rk = self.rv = None
        self.pos, self.t = [], 0

    def append(self, k, v):
        if self.t < self.n_sink:
            self.sk = k if self.sk is None else torch.cat([self.sk, k], 2)
            self.sv = v if self.sv is None else torch.cat([self.sv, v], 2)
        else:
            self.rk = k if self.rk is None else torch.cat([self.rk, k], 2)[:, :, -self.window:]
            self.rv = v if self.rv is None else torch.cat([self.rv, v], 2)[:, :, -self.window:]
        self.t += 1

    def get(self):
        parts_k = [p for p in (self.sk, self.rk) if p is not None]
        parts_v = [p for p in (self.sv, self.rv) if p is not None]
        return torch.cat(parts_k, 2), torch.cat(parts_v, 2)

    def positions(self):
        sinks = list(range(min(self.t, self.n_sink)))
        start = max(self.n_sink, self.t - self.window)
        return sinks + list(range(start, self.t))
