"""
title: Custom autograd Function — fake quantisation with a straight-through estimator
week: 5
day: 5
level: medium
time: 30
tags: pytorch, autograd, quantization
core: yes

Implement `FakeQuantSTE(torch.autograd.Function)` with `forward(ctx, x, scale, qmin, qmax)` returning
`clamp(round(x/scale), qmin, qmax) * scale` and a **clipped straight-through** `backward`: the gradient w.r.t. `x` is
the upstream gradient where `qmin*scale <= x <= qmax*scale`, and 0 outside; no gradient for the other arguments
(return `None`). Expose `fake_quant(x, scale, qmin=-128, qmax=127)`.

Then implement `RoundSTE` (plain round with identity gradient everywhere) and `round_ste(x)`.
"""

import torch

class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        raise NotImplementedError  # TODO: implement

    @staticmethod
    def backward(ctx, g):
        raise NotImplementedError  # TODO: implement


def fake_quant(x, scale, qmin=-128, qmax=127):
    raise NotImplementedError  # TODO: implement


class RoundSTE(torch.autograd.Function):
    @staticmethod
    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError  # TODO: implement

    @staticmethod
    def backward(ctx, g):
        raise NotImplementedError  # TODO: implement


def round_ste(x):
    raise NotImplementedError  # TODO: implement
