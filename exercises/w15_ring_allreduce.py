"""
title: Simulate ring all-reduce (reduce-scatter + all-gather) and count traffic
week: 15
day: 3
level: medium
time: 35
tags: numpy, distributed, collectives
core: yes

`ring_allreduce(buffers)` — `buffers` is a list of N equal-length 1-D NumPy arrays (one per simulated GPU). Split each
into N chunks (`np.array_split`). **Reduce-scatter:** for N−1 steps, rank r sends chunk `(r − step) mod N` to rank
`(r+1) mod N`, which adds it into its own copy. **All-gather:** for N−1 steps, pass the fully reduced chunks around the
ring. Return `(results, bytes_sent_per_rank)` where `results` are N arrays all equal to the element-wise sum and
`bytes_sent_per_rank` counts bytes each rank sent (`arr.nbytes` of the chunks). Do the sends of a step
"simultaneously" (copy what is sent before applying receives).
"""

import numpy as np

def ring_allreduce(buffers):
    raise NotImplementedError  # TODO: implement
