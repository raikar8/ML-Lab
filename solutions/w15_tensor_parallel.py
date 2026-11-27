"""
title: Megatron tensor parallelism for an MLP (simulated shards)
week: 15
day: 4
level: medium
time: 35
tags: pytorch, distributed, tensor-parallel
core: yes

The MLP is `y = gelu(x @ A) @ B` with `A (d, h)`, `B (h, d)`.

* `shard_mlp(A, B, world)` → list of `(A_i, B_i)`: `A` split by **columns**, `B` by **rows** into `world` equal parts.
* `tp_forward(x, shards)` — each "rank" computes `gelu(x @ A_i) @ B_i` locally (no communication), then a single
  **all-reduce (sum)** of the partial outputs gives `y`. Return `(y, n_allreduce)` with `n_allreduce == 1`.
* `bad_tp_forward(x, shards)` — the naive alternative: split `A` by **rows** (and `x` by columns), so each rank
  holds a partial pre-activation; you must all-reduce before the GeLU (communication #1), then split `B` by columns
  and all-gather the outputs (communication #2). Return `(y, 2)`. Same result, twice the communication — which is why
  Megatron uses column-then-row.
"""
import torch
import torch.nn.functional as F


def shard_mlp(A, B, world):
    return list(zip(A.chunk(world, dim=1), B.chunk(world, dim=0)))


def tp_forward(x, shards):
    partials = [F.gelu(x @ Ai) @ Bi for Ai, Bi in shards]
    return torch.stack(partials).sum(0), 1


def bad_tp_forward(x, shards):
    A = torch.cat([a for a, _ in shards], 1)
    B = torch.cat([b for _, b in shards], 0)
    world = len(shards)
    A_rows = A.chunk(world, 0)
    x_cols = x.chunk(world, -1)
    h = torch.stack([xi @ Ai for xi, Ai in zip(x_cols, A_rows)]).sum(0)   # all-reduce #1
    h = F.gelu(h)
    outs = [h @ Bc for Bc in B.chunk(world, 1)]
    return torch.cat(outs, -1), 2                                          # + all-gather #2
