"""
title: MobileNetV2 inverted residual block + parameter/MAC counter
week: 7
day: 3
level: medium
time: 35
tags: pytorch, efficient-cnn, embedded
core: yes

`InvertedResidual(c_in, c_out, stride, expand)`:
`1×1 conv (c_in→c_in·expand, no bias) → BN → ReLU6 → 3×3 depthwise conv (stride, pad 1, groups=hidden, no bias) → BN →
ReLU6 → 1×1 conv (→c_out, no bias) → BN` (no activation at the end), with a residual connection **only** when
`stride == 1 and c_in == c_out`. If `expand == 1`, skip the first 1×1 conv/BN/ReLU6. Expose `self.use_res`.

`count_macs(model, input_shape)` — total MACs of all `nn.Conv2d` and `nn.Linear` layers for one forward pass of a
tensor of `input_shape` (use forward hooks; conv MACs = `out.numel()/N * (C_in/groups) * k_h * k_w`).
`count_params(model)` — number of parameters.
"""

import torch

import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, stride, expand):
        raise NotImplementedError  # TODO: implement

    def forward(self, x):
        raise NotImplementedError  # TODO: implement


def count_params(model):
    raise NotImplementedError  # TODO: implement


def count_macs(model, input_shape):
    raise NotImplementedError  # TODO: implement
