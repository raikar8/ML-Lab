"""
title: Magnitude, Wanda and 2:4 pruning masks
week: 23
day: 2
level: medium
time: 35
tags: pytorch, pruning, llm
core: yes

`W (out, in)`, calibration inputs `X (n, in)`.

* `magnitude_mask(W, sparsity)` — keep the largest `|W|` **per output row**, pruning `round(sparsity·in)` weights per row.
* `wanda_mask(W, X, sparsity)` — same but score `|W_ij| · ‖X_:,j‖₂`.
* `nm_mask(scores, n=2, m=4)` — keep the top-n scores in each consecutive group of m along the input dim.
* `pruned_error(W, mask, X)` → `‖X Wᵀ − X (W·mask)ᵀ‖_F`.
"""

import torch

def row_topk_mask(scores, sparsity):
    raise NotImplementedError  # TODO: implement


def magnitude_mask(W, sparsity):
    raise NotImplementedError  # TODO: implement


def wanda_mask(W, X, sparsity):
    raise NotImplementedError  # TODO: implement


def nm_mask(scores, n=2, m=4):
    raise NotImplementedError  # TODO: implement


def pruned_error(W, mask, X):
    raise NotImplementedError  # TODO: implement
