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
    last = max(t[3] for t in tensors)
    return max(sum(t[1] for t in tensors if t[2] <= i <= t[3]) for i in range(last + 1))


def plan_arena(tensors, align=16):
    def up(v):
        return (v + align - 1) // align * align

    placed = []
    offsets = {}
    for name, size, a, b in sorted(tensors, key=lambda t: (-t[1], t[0])):
        conflicts = sorted((o, o + s) for (n, s, aa, bb, o) in placed if not (bb < a or b < aa))
        off = 0
        for lo, hi in conflicts:
            if off + size <= lo:
                break
            off = max(off, up(hi))
        offsets[name] = off
        placed.append((name, size, a, b, off))
    return offsets, max(o + s for (_, s, _, _, o) in placed)
