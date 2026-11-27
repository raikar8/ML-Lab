"""
title: LayerNorm, RMSNorm, BatchNorm1d from scratch + BN folding
week: 6
day: 3
level: medium
time: 40
tags: pytorch, normalisation, deployment
core: yes

Write `nn.Module`s that match PyTorch:

* `LayerNorm(d, eps=1e-5)` over the last dim with weight/bias (biased variance).
* `RMSNorm(d, eps=1e-6)`: `x * rsqrt(mean(x²) + eps) * weight` (compute in float32, cast back to the input dtype).
* `BatchNorm1d(C, eps=1e-5, momentum=0.1)` for inputs `(N, C)`: training uses batch mean and **biased** variance for
  normalising, and updates `running_mean`/`running_var` (buffers) with the **unbiased** variance:
  `running = (1−momentum)·running + momentum·batch`. Eval uses running stats.
* `fold_bn_into_linear(linear, bn)` → a new `nn.Linear` equal to `bn(linear(x))` in eval mode.
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class BatchNorm1d(nn.Module):
    def __init__(self, C, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.register_buffer("running_mean", torch.zeros(C))
        self.register_buffer("running_var", torch.ones(C))

    def forward(self, x):
        if self.training:
            mu = x.mean(0)
            var = x.var(0, unbiased=False)
            with torch.no_grad():
                n = x.shape[0]
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var * n / (n - 1))
        else:
            mu, var = self.running_mean, self.running_var
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


def fold_bn_into_linear(linear, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    out = nn.Linear(linear.in_features, linear.out_features)
    b = linear.bias if linear.bias is not None else torch.zeros(linear.out_features)
    with torch.no_grad():
        out.weight.copy_(linear.weight * s[:, None])
        out.bias.copy_((b - bn.running_mean) * s + bn.bias)
    return out
