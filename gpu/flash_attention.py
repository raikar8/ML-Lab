"""FlashAttention forward pass in Triton, written from scratch.

One program computes one block of BLOCK_M queries for one (batch, head): it streams over key/value blocks, keeping the
running max m, normaliser l and un-normalised output acc in registers (online softmax), so the T×T score matrix never
exists in HBM. Inputs are (B, H, T, D) contiguous, D a power of two ≤ 128.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd(q_ptr, k_ptr, v_ptr, o_ptr, T, scale,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    base = bh * T * D
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + base + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0).to(tl.float32)
    q = q * scale
    m = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    hi = tl.minimum((pid_m + 1) * BLOCK_M, T) if CAUSAL else T
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < T
        k = tl.load(k_ptr + base + offs_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + base + offs_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(k))
        valid = offs_n[None, :] < T
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= offs_m[:, None])
        s = tl.where(valid, s, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        corr = tl.exp(m - m_new)
        l = l * corr + tl.sum(p, axis=1)
        acc = acc * corr[:, None] + tl.dot(p, v)
        m = m_new
    out = acc / l[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * D + offs_d[None, :], out.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < T)


def flash_attention(q, k, v, causal=False, BLOCK_M=64, BLOCK_N=64):
    B, H, T, D = q.shape
    assert D & (D - 1) == 0 and D <= 128, "head dim must be a power of two ≤ 128"
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    grid = (triton.cdiv(T, BLOCK_M), B * H)
    _flash_fwd[grid](q, k, v, o, T, 1.0 / math.sqrt(D), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D, CAUSAL=causal)
    return o
