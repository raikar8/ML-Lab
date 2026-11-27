"""
title: Conv2d via im2col + GEMM
week: 7
day: 1
level: medium
time: 40
tags: pytorch, cnn, embedded
core: yes

Using only tensor ops (no `F.conv2d`, no `F.unfold`):

* `conv_out_size(n, k, stride=1, pad=0, dilation=1)` → int.
* `im2col(x, k, stride=1, pad=0)` — `x` is `(N, C, H, W)`; return `(N, C*k*k, H_out*W_out)` where column `p` holds the
  receptive field of output position `p` in `(c, ki, kj)` order (same layout as `F.unfold`).
* `conv2d(x, w, b=None, stride=1, pad=0)` — `w` is `(C_out, C, k, k)`; compute `w.view(C_out, -1) @ cols` and reshape.
* `conv_cost(C_in, C_out, k, H_out, W_out, groups=1)` → `(params_without_bias, MACs)`.
"""

import torch

def conv_out_size(n, k, stride=1, pad=0, dilation=1):
    raise NotImplementedError  # TODO: implement


def im2col(x, k, stride=1, pad=0):
    raise NotImplementedError  # TODO: implement


def conv2d(x, w, b=None, stride=1, pad=0):
    raise NotImplementedError  # TODO: implement


def conv_cost(C_in, C_out, k, H_out, W_out, groups=1):
    raise NotImplementedError  # TODO: implement
