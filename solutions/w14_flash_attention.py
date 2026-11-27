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
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    lse = torch.empty(B, H, T, dtype=q.dtype, device=q.device)
    for qs in range(0, T, block_q):
        qb = q[:, :, qs:qs + block_q] * scale
        nq = qb.shape[2]
        m = torch.full((B, H, nq), float("-inf"), dtype=q.dtype, device=q.device)
        l = torch.zeros(B, H, nq, dtype=q.dtype, device=q.device)
        acc = torch.zeros(B, H, nq, v.shape[-1], dtype=q.dtype, device=q.device)
        for ks in range(0, T, block_k):
            if causal and ks > qs + nq - 1:
                break
            kb, vb = k[:, :, ks:ks + block_k], v[:, :, ks:ks + block_k]
            s = qb @ kb.transpose(-1, -2)
            if causal:
                qi = torch.arange(qs, qs + nq, device=q.device)[:, None]
                ki = torch.arange(ks, ks + kb.shape[2], device=q.device)[None]
                s = s.masked_fill(ki > qi, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1))
            p = torch.exp(s - m_new[..., None])
            corr = torch.exp(m - m_new)
            l = l * corr + p.sum(-1)
            acc = acc * corr[..., None] + p @ vb
            m = m_new
        out[:, :, qs:qs + nq] = acc / l[..., None]
        lse[:, :, qs:qs + nq] = m + torch.log(l)
    return out, lse


def combine_partials(o1, lse1, o2, lse2):
    mx = torch.maximum(lse1, lse2)
    w1, w2 = torch.exp(lse1 - mx), torch.exp(lse2 - mx)
    o = (o1 * w1[..., None] + o2 * w2[..., None]) / (w1 + w2)[..., None]
    return o, mx + torch.log(w1 + w2)
