"""
title: Tensor-arena memory planner (TFLite-Micro / ExecuTorch style)
week: 20
day: 4
level: medium
time: 40
tags: python, embedded, memory
core: yes

Each tensor is `(name, size_bytes, first_use, last_use)` with inclusive op indices. Two tensors **conflict** if their
lifetimes overlap.

* `peak_live_bytes(tensors)` — max over op indices of the total size of tensors alive at that op (a lower bound).
* `plan_arena(tensors, align=16)` → `(offsets: dict name→offset, arena_size)`. Greedy: sort by size descending (ties by
  name); place each tensor at the lowest **aligned** offset where it doesn't overlap in memory with any already placed,
  lifetime-conflicting tensor. Arena size = max(offset + size).
"""

def peak_live_bytes(tensors):
    raise NotImplementedError  # TODO: implement


def plan_arena(tensors, align=16):
    raise NotImplementedError  # TODO: implement
