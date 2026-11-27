"""
title: LSQ fake-quant module for QAT (learned step size)
week: 21
day: 5
level: medium
time: 35
tags: pytorch, qat, quantization
core: yes

`LSQQuantizer(bits=4, signed=True, init_scale=None)` (an `nn.Module` with a learnable `scale` parameter):

* `qn, qp` = `−2^(b−1), 2^(b−1)−1` if signed else `0, 2^b−1`.
* On the first forward, if `init_scale` is None, initialise `scale = 2·mean|x| / sqrt(qp)` (LSQ paper).
* forward: gradient scale `g = 1/sqrt(x.numel()·qp)`; `s = grad_scale(scale, g)` (value unchanged, gradient × g);
  `x_q = round_ste(clamp(x/s, qn, qp)) · s` with a straight-through round.
* `QATLinear(in, out, bits)` — `nn.Linear` whose weight passes through an `LSQQuantizer` before the matmul.

Helpers to implement: `grad_scale(x, g)` (`(x − x·g).detach() + x·g`) and `round_ste(x)` (`(x.round() − x).detach() + x`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def grad_scale(x, g):
    return (x - x * g).detach() + x * g


def round_ste(x):
    return (x.round() - x).detach() + x


class LSQQuantizer(nn.Module):
    def __init__(self, bits=4, signed=True, init_scale=None):
        super().__init__()
        self.qn, self.qp = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2 ** bits - 1)
        self.scale = nn.Parameter(torch.tensor(float(init_scale) if init_scale else 1.0))
        self.initialised = init_scale is not None

    def forward(self, x):
        if not self.initialised:
            with torch.no_grad():
                self.scale.copy_(2 * x.abs().mean() / self.qp ** 0.5)
            self.initialised = True
        s = grad_scale(self.scale, 1.0 / (x.numel() * self.qp) ** 0.5)
        return round_ste(torch.clamp(x / s, self.qn, self.qp)) * s


class QATLinear(nn.Linear):
    def __init__(self, in_features, out_features, bits=4):
        super().__init__(in_features, out_features)
        self.wq = LSQQuantizer(bits)

    def forward(self, x):
        return F.linear(x, self.wq(self.weight), self.bias)
