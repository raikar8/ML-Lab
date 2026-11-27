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
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad.clone()
                if g["weight_decay"]:
                    d.add_(p, alpha=g["weight_decay"])
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = d.clone()
                else:
                    st["buf"].mul_(g["momentum"]).add_(d)
                p.add_(st["buf"], alpha=-g["lr"])


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["t"] += 1
                t = st["t"]
                p.mul_(1 - g["lr"] * g["weight_decay"])
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                mhat = st["m"] / (1 - b1 ** t)
                vhat = st["v"] / (1 - b2 ** t)
                p.addcdiv_(mhat, vhat.sqrt().add_(g["eps"]), value=-g["lr"])


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, "Muon is for 2-D weight matrices"
                st = self.state[p]
                buf = st.setdefault("buf", torch.zeros_like(p))
                buf.mul_(g["momentum"]).add_(p.grad)
                u = p.grad + g["momentum"] * buf if g["nesterov"] else buf
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(newton_schulz(u), alpha=-g["lr"] * scale)
