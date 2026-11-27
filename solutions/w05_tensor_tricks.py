"""
title: Tensor tricks — zero-copy framing, broadcasting, einsum, masks
week: 5
day: 4
level: easy
time: 25
tags: pytorch, tensors, broadcasting
core: yes

* `frame_signal(x, frame, hop)` — `x` is `(..., T)`; return a **view** `(..., n_frames, frame)` with no copy
  (use `unfold` or `as_strided`); drop the incomplete last frame.
* `pairwise_l2(a, b)` — `(B, N, D)` and `(B, M, D)` → `(B, N, M)` Euclidean distances via broadcasting/matmul.
* `attention_scores(q, k)` — `(B, H, Tq, D)`, `(B, H, Tk, D)` → `(B, H, Tq, Tk)` scaled by `1/sqrt(D)`, using `torch.einsum`.
* `causal_mask(T, device=None)` — bool `(T, T)` with **True where attention is allowed** (lower triangle incl. diagonal).
* `padding_mask(lengths, T)` — bool `(B, T)`, True for real (non-pad) positions.
"""
import math

import torch


def frame_signal(x, frame, hop):
    return x.unfold(-1, frame, hop)


def pairwise_l2(a, b):
    d2 = (a * a).sum(-1, keepdim=True) - 2 * a @ b.transpose(-1, -2) + (b * b).sum(-1).unsqueeze(-2)
    return d2.clamp_min(0).sqrt()


def attention_scores(q, k):
    return torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(q.shape[-1])


def causal_mask(T, device=None):
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def padding_mask(lengths, T):
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]
