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
    k_prune = int(round(sparsity * scores.shape[1]))
    mask = torch.ones_like(scores, dtype=torch.bool)
    if k_prune:
        idx = scores.topk(k_prune, dim=1, largest=False).indices
        mask.scatter_(1, idx, False)
    return mask


def magnitude_mask(W, sparsity):
    return row_topk_mask(W.abs(), sparsity)


def wanda_mask(W, X, sparsity):
    return row_topk_mask(W.abs() * X.norm(dim=0)[None], sparsity)


def nm_mask(scores, n=2, m=4):
    out, inp = scores.shape
    g = scores.reshape(out, inp // m, m)
    idx = g.topk(n, dim=-1).indices
    mask = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True)
    return mask.reshape(out, inp)


def pruned_error(W, mask, X):
    return float((X @ W.T - X @ (W * mask).T).norm())
