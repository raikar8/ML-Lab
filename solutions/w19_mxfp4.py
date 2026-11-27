"""
title: MXFP4 block quantization (OCP microscaling) vs INT4 group-wise
week: 19
day: 5
level: medium
time: 35
tags: pytorch, quantization, formats
core: yes

MXFP4: blocks of 32 values share one **power-of-two** scale (E8M0); each element is FP4 E2M1 with representable
magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.

* `FP4_GRID` — tensor of the 8 non-negative E2M1 values.
* `quant_e2m1(x)` — round each value to the nearest representable ±E2M1 value (ties: either is fine), saturating at ±6.
* `mxfp4(x, block=32)` — reshape the last dim into blocks; scale `2^(floor(log2(amax)) − 2)` (so the block max maps into
  [4, 8) and saturates at 6, as in the OCP spec's `emax = 2`); return `quant_e2m1(x/scale)·scale`.
* `int4_group(x, group=32)` — symmetric int4 per group: `s = amax/7`, values in [−7, 7].
* `bits_per_value(fmt)` — `"mxfp4"` → 4 + 8/32, `"int4_g32_fp16"` → 4 + 16/32.
"""
import torch

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quant_e2m1(x):
    mag = x.abs().clamp(max=6.0)
    idx = (mag[..., None] - FP4_GRID.to(x.dtype)).abs().argmin(-1)
    return torch.sign(x) * FP4_GRID.to(x.dtype)[idx]


def mxfp4(x, block=32):
    shp = x.shape
    b = x.reshape(*shp[:-1], -1, block)
    amax = b.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    scale = torch.exp2(torch.floor(torch.log2(amax)) - 2)
    return (quant_e2m1(b / scale) * scale).reshape(shp)


def int4_group(x, group=32):
    shp = x.shape
    g = x.reshape(*shp[:-1], -1, group)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 7
    return (torch.clamp(torch.round(g / s), -7, 7) * s).reshape(shp)


def bits_per_value(fmt):
    return {"mxfp4": 4 + 8 / 32, "int4_g32_fp16": 4 + 16 / 32}[fmt]
