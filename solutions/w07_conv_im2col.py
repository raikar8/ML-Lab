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
    return (n + 2 * pad - dilation * (k - 1) - 1) // stride + 1


def im2col(x, k, stride=1, pad=0):
    N, C, H, W = x.shape
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad))
    Ho, Wo = conv_out_size(H, k, stride, pad), conv_out_size(W, k, stride, pad)
    patches = x.unfold(2, k, stride).unfold(3, k, stride)        # N, C, Ho, Wo, k, k
    return patches.permute(0, 1, 4, 5, 2, 3).reshape(N, C * k * k, Ho * Wo)


def conv2d(x, w, b=None, stride=1, pad=0):
    N, C, H, W = x.shape
    Co, _, k, _ = w.shape
    cols = im2col(x, k, stride, pad)
    out = w.reshape(Co, -1) @ cols
    if b is not None:
        out = out + b[:, None]
    Ho, Wo = conv_out_size(H, k, stride, pad), conv_out_size(W, k, stride, pad)
    return out.reshape(N, Co, Ho, Wo)


def conv_cost(C_in, C_out, k, H_out, W_out, groups=1):
    params = C_out * (C_in // groups) * k * k
    return params, params * H_out * W_out
