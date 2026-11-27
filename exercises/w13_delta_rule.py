"""
title: Gated delta rule (Gated DeltaNet / KDA-style) and associative recall
week: 13
day: 3
level: hard
time: 45
tags: pytorch, delta-rule, linear-attention, chinese-models
core: yes

The state `S (B, H, Dv, Dk)` maps keys to values. For each step, with L2-normalised key `k_t`, learning rate
`beta_t ∈ (0,1)` and decay `alpha_t ∈ (0,1]`:

```
S_t = alpha_t · S_{t−1} (I − beta_t k_t k_tᵀ) + beta_t v_t k_tᵀ        o_t = S_t q_t
```

* `gated_delta_rule(q, k, v, beta, alpha=None)` — `q, k (B,H,T,Dk)`, `v (B,H,T,Dv)`, `beta (B,H,T)`, `alpha (B,H,T)` or
  `None` (=1, plain DeltaNet). Normalise `k` (and `q`) along the last dim inside. Return `o (B,H,T,Dv)`.
* `additive_linear_attention(q, k, v)` — the same but `S_t = S_{t−1} + v_t k_tᵀ` (no delta, no decay), for comparison.

The delta update is one LMS/NLMS step on `‖S k − v‖²` — it overwrites a key's old value instead of piling up.
"""

import torch

import torch.nn.functional as F

def gated_delta_rule(q, k, v, beta, alpha=None):
    raise NotImplementedError  # TODO: implement


def additive_linear_attention(q, k, v):
    raise NotImplementedError  # TODO: implement
