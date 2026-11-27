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
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.ctx = (k - 1) * dilation
        self.state = None

    def forward(self, x):
        return self.conv(F.pad(x, (self.ctx, 0)))

    def reset(self, batch_size):
        self.state = torch.zeros(batch_size, self.conv.in_channels, self.ctx,
                                 dtype=self.conv.weight.dtype, device=self.conv.weight.device)

    def step(self, chunk):
        if self.state is None:
            self.reset(chunk.shape[0])
        buf = torch.cat([self.state, chunk], dim=-1)
        out = self.conv(buf)
        self.state = buf[..., buf.shape[-1] - self.ctx:] if self.ctx else buf[..., :0]
        return out
