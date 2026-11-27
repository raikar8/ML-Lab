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
    s = torch.relu(torch.einsum("bhtd,bsd->bhts", qi, ki))
    I = torch.einsum("bht,bhts->bts", w, s)
    T = I.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool, device=I.device).tril()
    return I.masked_fill(~causal, float("-inf"))


def topk_sparse_attention(q, k, v, scores, topk):
    B, H, T, D = q.shape
    kk = min(topk, T)
    idx = scores.topk(kk, dim=-1).indices                         # (B, T, kk)
    keep = torch.zeros(B, T, T, dtype=torch.bool, device=q.device).scatter_(-1, idx, True)
    keep &= torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    logits = q @ k.transpose(-1, -2) / math.sqrt(D)
    logits = logits.masked_fill(~keep[:, None], float("-inf"))
    return torch.softmax(logits, -1) @ v
