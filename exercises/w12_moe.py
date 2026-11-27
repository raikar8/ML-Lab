"""
title: Top-k Mixture-of-Experts layer with shared expert and load-balancing loss
week: 12
day: 2
level: hard
time: 50
tags: pytorch, moe, transformers
core: yes

`MoE(d, hidden, n_experts, top_k, n_shared=0)`:

* experts: `n_experts` SwiGLU FFNs (`w1, w3: d→hidden`, `w2: hidden→d`, no bias) in an `nn.ModuleList` called `experts`;
  optional `shared` list of always-on experts of the same shape.
* router: `nn.Linear(d, n_experts, bias=False)` called `router`; softmax over experts, take top-k, **renormalise** the
  selected weights to sum to 1.
* `forward(x)` with `x (B, T, d)` → `(y, aux_loss)`. Dispatch tokens to experts by gathering only the tokens routed to
  each expert (no dense compute of all experts). `aux_loss = n_experts * Σ_i f_i · P_i` where `f_i` = fraction of
  (token, slot) assignments going to expert i (counts / (N·k)) and `P_i` = mean router probability of expert i.
* `self.last_load` — tensor `(n_experts,)` of token counts from the last forward.
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        raise NotImplementedError  # TODO: implement

    def forward(self, x):
        raise NotImplementedError  # TODO: implement


class MoE(nn.Module):
    def __init__(self, d, hidden, n_experts, top_k, n_shared=0):
        raise NotImplementedError  # TODO: implement

    def forward(self, x):
        raise NotImplementedError  # TODO: implement
