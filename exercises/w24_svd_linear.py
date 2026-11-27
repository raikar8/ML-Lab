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
    raise NotImplementedError  # TODO: implement


def svd_compress(linear, rank):
    raise NotImplementedError  # TODO: implement


def saves_params(in_f, out_f, rank):
    raise NotImplementedError  # TODO: implement
