"""
title: QK-clip (from Kimi K2's MuonClip)
week: 19
day: 2
level: medium
time: 25
tags: pytorch, stability, chinese-models
core: no

Per attention head h, with projection weights `Wq[h], Wk[h]` of shape `(d_head, d)` and a batch of inputs `x (T, d)`:

* `max_logits(Wq, Wk, x)` — `Wq, Wk (H, d_head, d)`; return `(H,)` with `max_{i,j} (Wq[h] x_i)·(Wk[h] x_j) / sqrt(d_head)`.
* `qk_clip_(Wq, Wk, x, tau)` — **in place**: for heads whose max logit exceeds `tau`, multiply `Wq[h]` and `Wk[h]` by
  `sqrt(tau / max_logit_h)` each (so the product scales by `tau/max`); leave other heads unchanged. Return the factors
  `(H,)` used (1 where not clipped).
"""

import math

import torch

def max_logits(Wq, Wk, x):
    raise NotImplementedError  # TODO: implement


@torch.no_grad()
def qk_clip_(Wq, Wk, x, tau):
    raise NotImplementedError  # TODO: implement
