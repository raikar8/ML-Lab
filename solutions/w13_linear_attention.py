"""
title: Causal linear attention — parallel, recurrent and chunkwise forms
week: 13
day: 2
level: hard
time: 45
tags: pytorch, linear-attention, efficient-attention
core: yes

Unnormalised causal linear attention with feature map `phi(x) = elu(x) + 1`: `o_t = Σ_{s≤t} (φ(q_t)·φ(k_s)) v_s`,
normalised by `φ(q_t)·Σ_{s≤t} φ(k_s)`. Shapes `(B, H, T, D)`.

* `linear_attention_parallel(q, k, v)` — materialise the `(T, T)` causal score matrix (O(T²)).
* `linear_attention_recurrent(q, k, v)` — loop over time with state `S (B,H,D,Dv)` and normaliser `z (B,H,D)`
  (O(T), constant memory per step).
* `linear_attention_chunked(q, k, v, chunk)` — within each chunk use the parallel form; carry `S, z` between chunks.
All three must agree.
"""
import torch
import torch.nn.functional as F


def phi(x):
    return F.elu(x) + 1


def linear_attention_parallel(q, k, v):
    q, k = phi(q), phi(k)
    A = (q @ k.transpose(-1, -2)).tril()
    return (A @ v) / A.sum(-1, keepdim=True)


def linear_attention_recurrent(q, k, v):
    q, k = phi(q), phi(k)
    B, H, T, D = q.shape
    S = q.new_zeros(B, H, D, v.shape[-1])
    z = q.new_zeros(B, H, D)
    outs = []
    for t in range(T):
        S = S + k[:, :, t, :, None] * v[:, :, t, None, :]
        z = z + k[:, :, t]
        num = (q[:, :, t, :, None] * S).sum(-2)
        den = (q[:, :, t] * z).sum(-1, keepdim=True)
        outs.append(num / den)
    return torch.stack(outs, 2)


def linear_attention_chunked(q, k, v, chunk):
    q, k = phi(q), phi(k)
    B, H, T, D = q.shape
    S = q.new_zeros(B, H, D, v.shape[-1])
    z = q.new_zeros(B, H, D)
    outs = []
    for s in range(0, T, chunk):
        qc, kc, vc = q[:, :, s:s + chunk], k[:, :, s:s + chunk], v[:, :, s:s + chunk]
        A = (qc @ kc.transpose(-1, -2)).tril()
        num = A @ vc + qc @ S
        den = A.sum(-1, keepdim=True) + (qc @ z[..., None])
        outs.append(num / den)
        S = S + kc.transpose(-1, -2) @ vc
        z = z + kc.sum(2)
    return torch.cat(outs, 2)
