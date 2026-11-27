"""
title: Optimisers from scratch — SGD+momentum, AdamW, and Muon (Newton–Schulz)
week: 6
day: 1
level: hard
time: 50
tags: pytorch, optimisers, chinese-models
core: yes

Subclass `torch.optim.Optimizer` (keep per-parameter state in `self.state[p]`, update in-place under `torch.no_grad()`):

1. `SGDMomentum(params, lr, momentum=0.9, weight_decay=0.0)` — match `torch.optim.SGD(momentum=..., weight_decay=...)`
   (buffer `b = μ b + g`, with `g += wd * p` first; first step `b = g`).
2. `AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)` — match `torch.optim.AdamW`
   (decoupled decay `p *= 1 − lr·wd`, bias-corrected moments).
3. `newton_schulz(G, steps=5)` — approximately orthogonalise a 2-D matrix with the quintic iteration
   `X ← aX + (bA + cA²)X`, `A = X Xᵀ`, `(a, b, c) = (3.4445, −4.7750, 2.0315)`, starting from `G/‖G‖_F`
   (transpose first if rows > cols, and back at the end).
4. `Muon(params, lr=0.02, momentum=0.95, nesterov=True)` — for 2-D params only: `b = μ b + g`; update direction
   `u = g + μ b` if nesterov else `b`; `p -= lr * max(1, rows/cols)**0.5 * newton_schulz(u)`.
"""

import torch

class SGDMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0):
        raise NotImplementedError  # TODO: implement

    @torch.no_grad()
    def step(self):
        raise NotImplementedError  # TODO: implement


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        raise NotImplementedError  # TODO: implement

    @torch.no_grad()
    def step(self):
        raise NotImplementedError  # TODO: implement


def newton_schulz(G, steps=5):
    raise NotImplementedError  # TODO: implement


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        raise NotImplementedError  # TODO: implement

    @torch.no_grad()
    def step(self):
        raise NotImplementedError  # TODO: implement
