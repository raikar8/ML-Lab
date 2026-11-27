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
    def forward(ctx, x, scale, qmin, qmax):
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = qmin * scale, qmax * scale
        return torch.clamp(torch.round(x / scale), qmin, qmax) * scale

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        mask = (x >= ctx.lo) & (x <= ctx.hi)
        return g * mask, None, None, None


def fake_quant(x, scale, qmin=-128, qmax=127):
    return FakeQuantSTE.apply(x, scale, qmin, qmax)


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def round_ste(x):
    return RoundSTE.apply(x)
