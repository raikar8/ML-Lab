"""
title: LoRA linear layer with merge/unmerge
week: 17
day: 2
level: easy
time: 25
tags: pytorch, peft, fine-tuning
core: yes

`LoRALinear(base: nn.Linear, r=8, alpha=16)`:
* freeze `base` (no grad); create `A (r, in)` initialised with `kaiming_uniform_(a=sqrt(5))` and `B (out, r)` = zeros;
  `scaling = alpha / r`.
* `forward(x) = base(x) + scaling · (x @ Aᵀ) @ Bᵀ` (skip the LoRA path when merged).
* `merge()` adds `scaling · B @ A` into `base.weight` and sets `merged = True`; `unmerge()` reverses it.
* `lora_parameters(model)` — list of all A/B parameters in a model (for the optimiser).
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scaling = alpha / r
        self.merged = False

    def forward(self, x):
        y = self.base(x)
        if not self.merged:
            y = y + self.scaling * (x @ self.A.T) @ self.B.T
        return y

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.scaling * self.B @ self.A
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.scaling * self.B @ self.A
            self.merged = False


def lora_parameters(model):
    return [p for mod in model.modules() if isinstance(mod, LoRALinear) for p in (mod.A, mod.B)]
