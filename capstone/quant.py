"""BN folding, int8 quantisation-aware training, and an integer-only inference engine (NumPy), all from scratch.

The integer engine computes exactly what an MCU kernel library (CMSIS-NN style) would: int8 activations with zero-points,
per-output-channel int8 weights, int32 accumulators and fixed-point requantisation (multiplier M0 in Q31 + right shift).
`export_c_header` writes the weights and requantisation constants as C arrays for an embedded build.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DSCNN


# ------------------------------------------------------------------ BN folding: DSCNN -> list of conv layers + fc
class FoldedNet(nn.Module):
    """The DSCNN with every BatchNorm folded into its convolution (inference graph), plus optional fake quantisation."""

    def __init__(self, model: DSCNN):
        super().__init__()
        self.convs = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.meta = []  # (stride, pad, groups)
        cbrs = [model.stem] + [m for b in model.blocks for m in b]
        for cbr in cbrs:
            conv, bn = cbr[0], cbr[1]
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            self.convs.append(nn.Parameter((conv.weight * s[:, None, None, None]).detach().clone()))
            self.biases.append(nn.Parameter((bn.bias - bn.running_mean * s).detach().clone()))
            self.meta.append((conv.stride[0], conv.padding[0], conv.groups))
        self.fc_w = nn.Parameter(model.fc.weight.detach().clone())
        self.fc_b = nn.Parameter(model.fc.bias.detach().clone())
        n_act = len(cbrs) + 1                       # input + one per conv output
        self.register_buffer("act_min", torch.zeros(n_act))
        self.register_buffer("act_max", torch.zeros(n_act))
        self.observing, self.quant = False, False

    def observe(self, i, x):
        if self.observing:
            lo, hi = x.detach().min(), x.detach().max()
            first = self.act_max[i] == 0 and self.act_min[i] == 0
            self.act_min[i] = lo if first else 0.9 * self.act_min[i] + 0.1 * lo  # EMA of batch min/max
            self.act_max[i] = hi if first else 0.9 * self.act_max[i] + 0.1 * hi

    def act_qparams(self, i):
        lo, hi = float(min(self.act_min[i], 0.0)), float(max(self.act_max[i], 1e-6))
        scale = (hi - lo) / 255.0
        zp = int(np.clip(round(-128 - lo / scale), -128, 127))
        return scale, zp

    def fq_act(self, i, x):
        if not self.quant:
            return x
        s, zp = self.act_qparams(i)
        q = torch.clamp(torch.round(x / s) + zp, -128, 127)
        return x + ((q - zp) * s - x).detach()          # straight-through estimator

    @staticmethod
    def fq_weight(w):
        s = w.detach().abs().flatten(1).amax(1).clamp_min(1e-8) / 127
        shape = (-1,) + (1,) * (w.ndim - 1)
        q = torch.clamp(torch.round(w / s.view(shape)), -127, 127) * s.view(shape)
        return w + (q - w).detach()

    def forward(self, x):
        a = x[:, None]
        self.observe(0, a)
        a = self.fq_act(0, a)
        for i, (w, b, (st, pad, g)) in enumerate(zip(self.convs, self.biases, self.meta)):
            a = F.relu(F.conv2d(a, self.fq_weight(w) if self.quant else w, b, st, pad, 1, g))
            self.observe(i + 1, a)
            a = self.fq_act(i + 1, a)
        a = a.mean((2, 3))
        if self.quant:  # integer average pooling keeps the input scale; emulate its rounding
            s, zp = self.act_qparams(len(self.convs))
            a = a + (torch.round(a / s) * s - a).detach()
        return F.linear(a, self.fq_weight(self.fc_w) if self.quant else self.fc_w, self.fc_b)


@torch.no_grad()
def calibrate(net: FoldedNet, x, batch=256):
    net.eval()
    net.observing, net.quant = True, False
    net.act_min.zero_()
    net.act_max.zero_()
    for i in range(0, len(x), batch):
        net(x[i:i + batch])
    net.observing = False


# ------------------------------------------------------------------ fixed-point requantisation
def quantize_multiplier(M: float):
    m, e = math.frexp(M)
    M0 = int(round(m * (1 << 31)))
    if M0 == 1 << 31:
        M0, e = M0 // 2, e + 1
    return M0, -e


def requantize(acc: np.ndarray, M0: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """round(acc * M0 / 2^31) then rounding arithmetic shift right; per-channel M0/shift broadcast on the last axis."""
    acc = acc.astype(np.int64)
    hi = (acc * M0 + (1 << 30)) >> 31
    r = np.where(shift > 0, np.left_shift(np.int64(1), np.maximum(shift - 1, 0)), 0)
    right = np.where(hi >= 0, (hi + r) >> np.maximum(shift, 0), -((-hi + r) >> np.maximum(shift, 0)))
    return np.where(shift > 0, right, np.left_shift(hi, np.maximum(-shift, 0)))  # M ≥ 1 gives a left shift


# ------------------------------------------------------------------ integer-only engine
class Int8Engine:
    """Built from a calibrated FoldedNet. `run(x)` returns float logits computed with integer arithmetic only
    (except the final dequantisation of the logits for reporting)."""

    def __init__(self, net: FoldedNet):
        self.layers = []
        s_in, zp_in = net.act_qparams(0)
        self.s_in, self.zp_in = s_in, zp_in
        s_x, zp_x = s_in, zp_in
        for i, (w, b, (st, pad, g)) in enumerate(zip(net.convs, net.biases, net.meta)):
            w = w.detach().double().numpy()
            sw = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(1), 1e-8) / 127
            qw = np.clip(np.round(w / sw[:, None, None, None]), -127, 127).astype(np.int8)
            s_y, zp_y = net.act_qparams(i + 1)
            qb = np.round(b.detach().double().numpy() / (s_x * sw)).astype(np.int64)
            mult = [quantize_multiplier(s_x * swc / s_y) for swc in sw]
            self.layers.append(dict(kind="conv", qw=qw, qb=qb, M0=np.array([m for m, _ in mult], np.int64),
                                    shift=np.array([s for _, s in mult], np.int64), zp_x=zp_x, zp_y=zp_y,
                                    stride=st, pad=pad, groups=g, s_y=s_y))
            s_x, zp_x = s_y, zp_y
        w = net.fc_w.detach().double().numpy()
        sw = np.maximum(np.abs(w).max(1), 1e-8) / 127
        self.fc = dict(qw=np.clip(np.round(w / sw[:, None]), -127, 127).astype(np.int8),
                       qb=np.round(net.fc_b.detach().double().numpy() / (s_x * sw)).astype(np.int64),
                       s_out=s_x * sw, zp_x=zp_x)

    @staticmethod
    def conv(q, L):
        """q (C, H, W) int8 -> (Co, Ho, Wo) int8. im2col with the input zero-point as padding value."""
        C, H, W = q.shape
        qw, k, st, p, g = L["qw"], L["qw"].shape[-1], L["stride"], L["pad"], L["groups"]
        x = np.pad(q.astype(np.int64) - L["zp_x"], ((0, 0), (p, p), (p, p)))   # padded real zeros = (zp - zp) = 0
        Ho, Wo = (H + 2 * p - k) // st + 1, (W + 2 * p - k) // st + 1
        Co, cpg = qw.shape[0], C // g
        out = np.empty((Co, Ho * Wo), np.int64)
        opg = Co // g
        for gi in range(g):
            xs = x[gi * cpg:(gi + 1) * cpg]
            cols = np.stack([xs[:, i:i + st * Ho:st, j:j + st * Wo:st] for i in range(k) for j in range(k)], 1)  # cpg,k*k,Ho,Wo
            cols = cols.reshape(cpg * k * k, Ho * Wo)
            wg = qw[gi * opg:(gi + 1) * opg].reshape(opg, -1).astype(np.int64)
            out[gi * opg:(gi + 1) * opg] = wg @ cols
        acc = out + L["qb"][:, None]
        y = L["zp_y"] + requantize(acc.T, L["M0"], L["shift"]).T
        return np.clip(y, L["zp_y"], 127).astype(np.int8).reshape(Co, Ho, Wo)   # lower clamp at zp = fused ReLU

    def run_one(self, x):
        q = np.clip(np.round(x / self.s_in) + self.zp_in, -128, 127).astype(np.int8)[None]
        for L in self.layers:
            q = self.conv(q, L)
        zp = self.layers[-1]["zp_y"]
        tot = (q.astype(np.int64) - zp).reshape(q.shape[0], -1)
        n = tot.shape[1]
        s = tot.sum(1)
        pooled = np.where(s >= 0, (s + n // 2) // n, -((-s + n // 2) // n))          # rounded integer mean, same scale
        acc = self.fc["qw"].astype(np.int64) @ pooled + self.fc["qb"]
        return acc * self.fc["s_out"]

    def run(self, x):
        return np.stack([self.run_one(xi) for xi in np.asarray(x, dtype=np.float64)])

    def size_bytes(self):
        return sum(L["qw"].size + 4 * L["qb"].size + 8 * L["M0"].size for L in self.layers) + self.fc["qw"].size + 4 * self.fc["qb"].size

    def peak_activation_bytes(self, shape=(40, 98)):
        h, w, c, peak = shape[0], shape[1], 1, 0
        for L in self.layers:
            k, st, p = L["qw"].shape[-1], L["stride"], L["pad"]
            ho, wo = (h + 2 * p - k) // st + 1, (w + 2 * p - k) // st + 1
            peak = max(peak, c * h * w + L["qw"].shape[0] * ho * wo)   # input + output tensors live together
            h, w, c = ho, wo, L["qw"].shape[0]
        return peak


def export_c_header(engine: Int8Engine, path, n_classes):
    """Write weights and requantisation constants as C arrays (for a CMSIS-NN-style or hand-written MCU runtime —
    the C++ track's int8 GEMM, im2col and requantisation exercises are the building blocks)."""
    def arr(ctype, name, a):
        vals = ", ".join(str(int(v)) for v in np.asarray(a).ravel())
        return f"static const {ctype} {name}[{np.asarray(a).size}] = {{{vals}}};\n"
    out = ["// Generated by capstone/pipeline.py — int8 DS-CNN keyword spotter\n#pragma once\n#include <stdint.h>\n\n",
           f"#define KWS_N_LAYERS {len(engine.layers)}\n#define KWS_N_CLASSES {n_classes}\n",
           f"#define KWS_INPUT_SCALE {engine.s_in:.9g}f\n#define KWS_INPUT_ZP {engine.zp_in}\n\n"]
    for i, L in enumerate(engine.layers):
        out.append(f"// layer {i}: conv {tuple(L['qw'].shape)} stride {L['stride']} pad {L['pad']} groups {L['groups']} "
                   f"zp_x {L['zp_x']} zp_y {L['zp_y']}\n")
        out += [arr("int8_t", f"L{i}_W", L["qw"]), arr("int32_t", f"L{i}_B", L["qb"]),
                arr("int32_t", f"L{i}_M0", L["M0"]), arr("int8_t", f"L{i}_SHIFT", L["shift"])]
    out += [arr("int8_t", "FC_W", engine.fc["qw"]), arr("int32_t", "FC_B", engine.fc["qb"])]
    with open(path, "w") as f:
        f.writelines(out)
