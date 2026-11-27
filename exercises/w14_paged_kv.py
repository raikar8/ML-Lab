"""
title: Paged KV cache with block tables and copy-on-write prefix sharing
week: 14
day: 2
level: hard
time: 45
tags: pytorch, kv-cache, serving, inference
core: no

`PagedKVCache(num_blocks, block_size, n_heads, head_dim)` holds a physical pool `k_pool, v_pool` of shape
`(num_blocks, block_size, n_heads, head_dim)` and a free list.

* `new_seq(seq_id)`; `append(seq_id, k, v)` — `k, v (n_heads, head_dim)` for one token; allocate a new block when the
  last one is full. Raise `MemoryError` when no free block remains.
* `gather(seq_id)` → `(k, v)` each `(T, n_heads, head_dim)` in token order, via the block table.
* `fork(src_id, dst_id)` — the new sequence **shares** the source's blocks (increase ref counts). Before appending into a
  shared, partially filled last block, copy it (copy-on-write).
* `free(seq_id)` — decrement ref counts, return blocks with count 0 to the free list. `num_free()` → int.
"""

import torch

class PagedKVCache:
    def __init__(self, num_blocks, block_size, n_heads, head_dim):
        raise NotImplementedError  # TODO: implement

    def num_free(self):
        raise NotImplementedError  # TODO: implement

    def alloc(self):
        raise NotImplementedError  # TODO: implement

    def new_seq(self, seq_id):
        raise NotImplementedError  # TODO: implement

    def append(self, seq_id, k, v):
        raise NotImplementedError  # TODO: implement

    def gather(self, seq_id):
        raise NotImplementedError  # TODO: implement

    def fork(self, src_id, dst_id):
        raise NotImplementedError  # TODO: implement

    def free(self, seq_id):
        raise NotImplementedError  # TODO: implement
