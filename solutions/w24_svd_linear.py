"""
title: Compress a Linear layer with truncated SVD
week: 24
day: 5
level: easy
time: 25
tags: pytorch, low-rank, compression
core: no

* `rank_for_energy(W, energy)` — smallest r whose top singular values keep at least `energy` of `Σσ²`.
* `svd_compress(linear, rank)` → `nn.Sequential(nn.Linear(in, rank, bias=False), nn.Linear(rank, out, bias=True))`
  whose product is the best rank-r approximation of `linear.weight` (split `√Σ` between the two factors); copy the bias.
* `saves_params(in_f, out_f, rank)` → True if `rank·(in+out) < in·out`.
"""
import torch
import torch.nn as nn


def rank_for_energy(W, energy):
    s = torch.linalg.svdvals(W)
    frac = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
    return int((frac < energy - 1e-12).sum().item() + 1)


def svd_compress(linear, rank):
    U, S, Vh = torch.linalg.svd(linear.weight.detach(), full_matrices=False)
    r = S[:rank].sqrt()
    a = nn.Linear(linear.in_features, rank, bias=False)
    b = nn.Linear(rank, linear.out_features, bias=True)
    with torch.no_grad():
        a.weight.copy_(r[:, None] * Vh[:rank])
        b.weight.copy_(U[:, :rank] * r[None])
        b.bias.copy_(linear.bias if linear.bias is not None else torch.zeros(linear.out_features))
    return nn.Sequential(a, b)


def saves_params(in_f, out_f, rank):
    return rank * (in_f + out_f) < in_f * out_f
