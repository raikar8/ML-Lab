"""
title: Streaming causal dilated Conv1d with state buffers
week: 7
day: 4
level: medium
time: 35
tags: pytorch, audio, streaming, real-time
core: yes

`StreamingCausalConv1d(c_in, c_out, k, dilation=1)` wraps an `nn.Conv1d` (stride 1, **no padding**):

* `forward(x)` — offline causal conv on `(B, C_in, T)`: left-pad by `(k-1)*dilation` zeros → output `(B, C_out, T)`.
* `reset(batch_size)` — zero state of shape `(B, C_in, (k-1)*dilation)`.
* `step(chunk)` — chunk `(B, C_in, t)` of any length `t ≥ 1`; concatenate state + chunk, run the conv, keep the last
  `(k-1)*dilation` input frames as the new state, return `(B, C_out, t)`.

Streaming over arbitrary chunk sizes must equal `forward` on the whole signal. Handle `k == 1` (empty state).
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

class StreamingCausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, k, dilation=1):
        raise NotImplementedError  # TODO: implement

    def forward(self, x):
        raise NotImplementedError  # TODO: implement

    def reset(self, batch_size):
        raise NotImplementedError  # TODO: implement

    def step(self, chunk):
        raise NotImplementedError  # TODO: implement
