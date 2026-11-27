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
    raise NotImplementedError  # TODO: implement


def pairwise_l2(a, b):
    raise NotImplementedError  # TODO: implement


def attention_scores(q, k):
    raise NotImplementedError  # TODO: implement


def causal_mask(T, device=None):
    raise NotImplementedError  # TODO: implement


def padding_mask(lengths, T):
    raise NotImplementedError  # TODO: implement
