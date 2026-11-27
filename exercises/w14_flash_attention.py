"""
title: FlashAttention forward — tiled attention with the online softmax
week: 14
day: 3
level: hard
time: 45
tags: pytorch, kernels, attention, inference
core: yes

Implement `flash_attention(q, k, v, block_q=16, block_k=16, causal=False)` for `(B, H, T, D)` tensors **without ever
forming the full `(T, T)` score matrix**: loop over query blocks; for each, loop over key/value blocks keeping a running
row max `m`, running normaliser `l` and un-normalised output `acc`; when a new block raises the max, rescale `l` and
`acc` by `exp(m_old − m_new)`. For `causal=True`, skip key blocks entirely in the future and mask within the diagonal
block. Also return the per-row log-sum-exp `lse = m + log(l)` (`(B, H, T)`), which the backward pass would use.

* `combine_partials(o1, lse1, o2, lse2)` — merge two partial attention results over disjoint key sets (this is how
  FlashDecoding merges split-KV chunks).
"""

import math

import torch

def flash_attention(q, k, v, block_q=16, block_k=16, causal=False):
    raise NotImplementedError  # TODO: implement


def combine_partials(o1, lse1, o2, lse2):
    raise NotImplementedError  # TODO: implement
