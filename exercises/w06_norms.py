"""
title: LayerNorm, RMSNorm, BatchNorm1d from scratch + BN folding
week: 6
day: 3
level: medium
time: 40
tags: pytorch, normalisation, deployment
core: yes

Write `nn.Module`s that match PyTorch:

* `LayerNorm(d, eps=1e-5)` over the last dim with weight/bias (biased variance).
* `RMSNorm(d, eps=1e-6)`: `x * rsqrt(mean(x²) + eps) * weight` (compute in float32, cast back to the input dtype).
* `BatchNorm1d(C, eps=1e-5, momentum=0.1)` for inputs `(N, C)`: training uses batch mean and **biased** variance for
  normalising, and updates `running_mean`/`running_var` (buffers) with the **unbiased** variance:
  `running = (1−momentum)·running + momentum·batch`. Eval uses running stats.
* `fold_bn_into_linear(linear, bn)` → a new `nn.Linear` equal to `bn(linear(x))` in eval mode.
"""

import torch

import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        raise NotImplementedError  # TODO: implement

    def forward(self, x):
        raise NotImplementedError  # TODO: implement


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        raise NotImplementedError  # TODO: implement

    def forward(self, x):
        raise NotImplementedError  # TODO: implement


class BatchNorm1d(nn.Module):
    def __init__(self, C, eps=1e-5, momentum=0.1):
        raise NotImplementedError  # TODO: implement

    def forward(self, x):
        raise NotImplementedError  # TODO: implement


def fold_bn_into_linear(linear, bn):
    raise NotImplementedError  # TODO: implement
