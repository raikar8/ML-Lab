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
        raise NotImplementedError  # TODO: implement

    def forward(self, x):
        raise NotImplementedError  # TODO: implement

    @torch.no_grad()
    def merge(self):
        raise NotImplementedError  # TODO: implement

    @torch.no_grad()
    def unmerge(self):
        raise NotImplementedError  # TODO: implement


def lora_parameters(model):
    raise NotImplementedError  # TODO: implement
