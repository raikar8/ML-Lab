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
        self.bs = block_size
        self.k_pool = torch.zeros(num_blocks, block_size, n_heads, head_dim)
        self.v_pool = torch.zeros_like(self.k_pool)
        self.free_list = list(range(num_blocks))[::-1]
        self.ref = [0] * num_blocks
        self.tables, self.lens = {}, {}

    def num_free(self):
        return len(self.free_list)

    def alloc(self):
        if not self.free_list:
            raise MemoryError("KV cache full")
        b = self.free_list.pop()
        self.ref[b] = 1
        return b

    def new_seq(self, seq_id):
        self.tables[seq_id], self.lens[seq_id] = [], 0

    def append(self, seq_id, k, v):
        tab, n = self.tables[seq_id], self.lens[seq_id]
        off = n % self.bs
        if off == 0:
            tab.append(self.alloc())
        elif self.ref[tab[-1]] > 1:
            old = tab[-1]
            new = self.alloc()
            self.k_pool[new], self.v_pool[new] = self.k_pool[old], self.v_pool[old]
            self.ref[old] -= 1
            tab[-1] = new
        b = tab[-1]
        self.k_pool[b, off], self.v_pool[b, off] = k, v
        self.lens[seq_id] = n + 1

    def gather(self, seq_id):
        tab, n = self.tables[seq_id], self.lens[seq_id]
        k = self.k_pool[tab].reshape(-1, *self.k_pool.shape[2:])[:n]
        v = self.v_pool[tab].reshape(-1, *self.v_pool.shape[2:])[:n]
        return k, v

    def fork(self, src_id, dst_id):
        self.tables[dst_id] = list(self.tables[src_id])
        self.lens[dst_id] = self.lens[src_id]
        for b in self.tables[dst_id]:
            self.ref[b] += 1

    def free(self, seq_id):
        for b in self.tables.pop(seq_id):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free_list.append(b)
        self.lens.pop(seq_id)
