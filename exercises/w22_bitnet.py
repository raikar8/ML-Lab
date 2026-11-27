"""
title: BitNet b1.58 ternary linear layer (absmean) with STE training
week: 22
day: 5
level: medium
time: 30
tags: pytorch, quantization, low-bit
core: no

* `ternarize(W)` → `(Wt, scale)`: `scale = mean|W|`, `Wt = clamp(round(W/scale), −1, 1)` (values in {−1, 0, 1}).
* `act_quant_int8(x)` — per-token absmax int8 fake-quant: `s = 127 / max|x|` per row, `round(x·s).clamp(−128,127)/s`.
* `BitLinear(in, out)` — `nn.Linear` without bias whose forward uses STE versions:
  `w_q = W + (Wt·scale − W).detach()`, `x_q = x + (act_quant_int8(x) − x).detach()`, returns `x_q @ w_qᵀ`.
"""

import torch

import torch.nn as nn

def ternarize(W):
    raise NotImplementedError  # TODO: implement


def act_quant_int8(x):
    raise NotImplementedError  # TODO: implement


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        raise NotImplementedError  # TODO: implement

    def forward(self, x):
        raise NotImplementedError  # TODO: implement
