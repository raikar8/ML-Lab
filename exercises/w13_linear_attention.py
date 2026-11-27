"""
title: Causal linear attention — parallel, recurrent and chunkwise forms
week: 13
day: 2
level: hard
time: 45
tags: pytorch, linear-attention, efficient-attention
core: yes

Unnormalised causal linear attention with feature map `phi(x) = elu(x) + 1`: `o_t = Σ_{s≤t} (φ(q_t)·φ(k_s)) v_s`,
normalised by `φ(q_t)·Σ_{s≤t} φ(k_s)`. Shapes `(B, H, T, D)`.

* `linear_attention_parallel(q, k, v)` — materialise the `(T, T)` causal score matrix (O(T²)).
* `linear_attention_recurrent(q, k, v)` — loop over time with state `S (B,H,D,Dv)` and normaliser `z (B,H,D)`
  (O(T), constant memory per step).
* `linear_attention_chunked(q, k, v, chunk)` — within each chunk use the parallel form; carry `S, z` between chunks.
All three must agree.
"""

import torch

import torch.nn.functional as F

def phi(x):
    raise NotImplementedError  # TODO: implement


def linear_attention_parallel(q, k, v):
    raise NotImplementedError  # TODO: implement


def linear_attention_recurrent(q, k, v):
    raise NotImplementedError  # TODO: implement


def linear_attention_chunked(q, k, v, chunk):
    raise NotImplementedError  # TODO: implement
