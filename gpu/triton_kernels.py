"""Triton kernels: fused softmax, fused residual-add + RMSNorm, tiled matmul, int8 weight-only matmul.

Each wrapper launches a kernel over a grid of programs; each program handles one tile ("block") of the output.
Run on CPU for testing with:  TRITON_INTERPRET=1 python -m pytest tests/test_gpu_kernels.py
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- fused row softmax: 1 read + 1 write per element
@triton.jit
def _softmax_kernel(x_ptr, y_ptr, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(y_ptr + row * stride_y + offs, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax of a 2-D tensor (one program per row; the whole row fits in one block)."""
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](x, y, n_cols, x.stride(0), y.stride(0), BLOCK=triton.next_power_of_2(n_cols))
    return y


# ---------------------------------------------------------------- fused residual add + RMSNorm
@triton.jit
def _add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, y_ptr, res_out_ptr, n_cols, eps, BLOCK: tl.constexpr, HAS_RES: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        x += tl.load(r_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(res_out_ptr + row * n_cols + offs, x, mask=mask)
    ms = tl.sum(x * x, axis=0) / n_cols
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + offs, x * tl.rsqrt(ms + eps) * w, mask=mask)


def add_rmsnorm(x, weight, residual=None, eps=1e-6):
    """Returns rmsnorm(x + residual) and (x + residual) — the pre-norm block pattern — in one pass over memory."""
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    y = torch.empty_like(x2)
    res_out = torch.empty_like(x2) if residual is not None else y
    r2 = residual.reshape(-1, x.shape[-1]).contiguous() if residual is not None else x2
    _add_rmsnorm_kernel[(x2.shape[0],)](x2, r2, weight, y, res_out, x2.shape[1], eps,
                                        BLOCK=triton.next_power_of_2(x2.shape[1]), HAS_RES=residual is not None)
    return y.view_as(x), (res_out.view_as(x) if residual is not None else None)


# ---------------------------------------------------------------- tiled matmul with fp32 accumulation
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * sam + (k0 + rk)[None, :] * sak,
                    mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn,
                    mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(c_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a, b, BM=64, BN=64, BK=32):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                         BM=BM, BN=BN, BK=BK)
    return c


# ---------------------------------------------------------------- int8 weight-only matmul (dequantise in registers)
@triton.jit
def _w8_matmul_kernel(a_ptr, w_ptr, s_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk)[None, :], mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        w = tl.load(w_ptr + (k0 + rk)[:, None] * N + rn[None, :], mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0)
        acc += tl.dot(a.to(tl.float32), w.to(tl.float32))
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], (acc * s[None, :]).to(c_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def quantize_w8(w):
    """Symmetric per-output-column int8 quantisation of a (K, N) weight. Returns (int8 weights, fp32 scales)."""
    s = w.abs().amax(0).clamp_min(1e-8).float() / 127
    return torch.clamp(torch.round(w.float() / s), -127, 127).to(torch.int8), s


def w8_matmul(a, w_q, scale, BM=32, BN=64, BK=32):
    """a (M, K) float  @  dequant(w_q (K, N) int8, scale (N,))  — weights stay int8 in memory (half the bytes of fp16)."""
    a = a.contiguous()
    M, K = a.shape
    N = w_q.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    _w8_matmul_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, w_q.contiguous(), scale.contiguous(), c, M, N, K,
                                                                  BM=BM, BN=BN, BK=BK)
    return c
