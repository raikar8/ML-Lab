"""
title: Indexer-based top-k sparse attention (DSA-style)
week: 13
day: 4
level: hard
time: 40
tags: pytorch, sparse-attention, chinese-models
core: no

A lightweight **indexer** scores every (query, key) pair; the main attention only looks at the top-k keys per query.

* `indexer_scores(qi, ki, w)` — indexer queries `qi (B, Hi, T, Di)`, keys `ki (B, T, Di)` (shared across indexer heads),
  per-head weights `w (B, Hi, T)`; score `I[b,t,s] = Σ_h w[b,h,t] · relu(qi[b,h,t] · ki[b,s])`, causal (−inf for `s > t`).
  Returns `(B, T, T)`.
* `topk_sparse_attention(q, k, v, scores, topk)` — main attention `(B, H, T, D)`; for each query keep only its `topk`
  highest-scoring causal keys (all of them if fewer are available), masked softmax attention over those.
With `topk ≥ T` it must equal dense causal attention.
"""

import math

import torch

def indexer_scores(qi, ki, w):
    raise NotImplementedError  # TODO: implement


def topk_sparse_attention(q, k, v, scores, topk):
    raise NotImplementedError  # TODO: implement
