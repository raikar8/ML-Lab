"""
title: InfoNCE / CLIP symmetric contrastive loss
week: 16
day: 5
level: easy
time: 20
tags: pytorch, contrastive, ssl
core: yes

`clip_loss(a, b, temperature=0.07)` — `a, b (N, D)` embeddings of N positive pairs (row i of `a` matches row i of `b`).
L2-normalise both, compute logits `a @ b.T / temperature`, and return the mean of the cross-entropy over rows
(a→b) and over columns (b→a) with targets `arange(N)`.
"""
import torch
import torch.nn.functional as F


def clip_loss(a, b, temperature=0.07):
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    target = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
