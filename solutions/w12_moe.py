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
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, d, hidden, n_experts, top_k, n_shared=0):
        super().__init__()
        self.E, self.k = n_experts, top_k
        self.experts = nn.ModuleList([SwiGLU(d, hidden) for _ in range(n_experts)])
        self.shared = nn.ModuleList([SwiGLU(d, hidden) for _ in range(n_shared)])
        self.router = nn.Linear(d, n_experts, bias=False)
        self.last_load = torch.zeros(n_experts)

    def forward(self, x):
        B, T, d = x.shape
        xf = x.reshape(-1, d)
        probs = torch.softmax(self.router(xf).float(), -1)
        w, idx = probs.topk(self.k, dim=-1)
        w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
        y = torch.zeros_like(xf)
        for e in range(self.E):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                y.index_add_(0, tok, self.experts[e](xf[tok]) * w[tok, slot, None])
        for s in self.shared:
            y = y + s(xf)
        counts = torch.bincount(idx.flatten(), minlength=self.E).float()
        self.last_load = counts.detach()
        f = counts / idx.numel()
        aux = self.E * (f * probs.mean(0)).sum()
        return y.view(B, T, d), aux
