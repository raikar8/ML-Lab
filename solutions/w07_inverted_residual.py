"""
title: MobileNetV2 inverted residual block + parameter/MAC counter
week: 7
day: 3
level: medium
time: 35
tags: pytorch, efficient-cnn, embedded
core: yes

`InvertedResidual(c_in, c_out, stride, expand)`:
`1×1 conv (c_in→c_in·expand, no bias) → BN → ReLU6 → 3×3 depthwise conv (stride, pad 1, groups=hidden, no bias) → BN →
ReLU6 → 1×1 conv (→c_out, no bias) → BN` (no activation at the end), with a residual connection **only** when
`stride == 1 and c_in == c_out`. If `expand == 1`, skip the first 1×1 conv/BN/ReLU6. Expose `self.use_res`.

`count_macs(model, input_shape)` — total MACs of all `nn.Conv2d` and `nn.Linear` layers for one forward pass of a
tensor of `input_shape` (use forward hooks; conv MACs = `out.numel()/N * (C_in/groups) * k_h * k_w`).
`count_params(model)` — number of parameters.
"""
import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, stride, expand):
        super().__init__()
        hidden = c_in * expand
        self.use_res = stride == 1 and c_in == c_out
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(c_in, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden),
                   nn.ReLU6(inplace=True), nn.Conv2d(hidden, c_out, 1, bias=False), nn.BatchNorm2d(c_out)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, input_shape):
    total = [0]

    def hook(mod, inp, out):
        n = out.shape[0]
        if isinstance(mod, nn.Conv2d):
            kh, kw = mod.kernel_size
            total[0] += out.numel() // n * (mod.in_channels // mod.groups) * kh * kw
        elif isinstance(mod, nn.Linear):
            total[0] += out.numel() // n * mod.in_features

    hs = [mm.register_forward_hook(hook) for mm in model.modules() if isinstance(mm, (nn.Conv2d, nn.Linear))]
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    model.train(was)
    for h in hs:
        h.remove()
    return total[0]
