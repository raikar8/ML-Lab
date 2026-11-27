"""
title: Manifold-constrained hyper-connections — Sinkhorn projection and n-stream residuals
week: 11
day: 4
level: hard
time: 40
tags: pytorch, residual, chinese-models
core: no

DeepSeek's mHC mixes `n` residual streams with a **doubly stochastic** matrix.

* `sinkhorn(logits, iters=20)` — map a `(n, n)` (or batched `(…, n, n)`) real matrix to an approximately doubly
  stochastic one: `P = exp(logits)`, then alternately normalise rows and columns to sum to 1.
* `HyperConnection(n, d, layer)` — wraps a sub-layer `F: (B, T, d) → (B, T, d)`. State `x` has shape `(B, T, n, d)`.
  Parameters: `res_logits (n, n)` (init so the matrix starts near identity, e.g. `eye*4`), `pre (n,)` and `post (n,)`
  (init to `1/n` and `1`; pass them through `softplus`-free ReLU or keep them non-negative with `abs`). Forward:
  `h = Σ_i pre_i · x[:, :, i]`, `y = F(h)`, `x_new = H_res · x  +  post_i · y` per stream, with
  `H_res = sinkhorn(res_logits)` mixing the stream dimension.
* `expand(x, n)` / `collapse(x)` — `(B, T, d) → (B, T, n, d)` by copying, and back by averaging streams.
"""
import torch
import torch.nn as nn


def sinkhorn(logits, iters=20):
    P = torch.exp(logits - logits.amax(dim=(-1, -2), keepdim=True))
    for _ in range(iters):
        P = P / P.sum(-1, keepdim=True)
        P = P / P.sum(-2, keepdim=True)
    return P


def expand(x, n):
    return x.unsqueeze(2).expand(-1, -1, n, -1).clone()


def collapse(x):
    return x.mean(2)


class HyperConnection(nn.Module):
    def __init__(self, n, d, layer):
        super().__init__()
        self.n, self.layer = n, layer
        self.res_logits = nn.Parameter(torch.eye(n) * 4.0)
        self.pre = nn.Parameter(torch.full((n,), 1.0 / n))
        self.post = nn.Parameter(torch.ones(n))

    def forward(self, x):
        H = sinkhorn(self.res_logits)
        h = torch.einsum("i,btid->btd", self.pre.abs(), x)
        y = self.layer(h)
        return torch.einsum("ij,btjd->btid", H, x) + self.post.abs()[None, None, :, None] * y.unsqueeze(2)
