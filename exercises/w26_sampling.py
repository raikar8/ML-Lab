"""
title: Decoding — temperature, top-k, top-p, repetition penalty, and a sampler
week: 26
day: 2
level: medium
time: 30
tags: pytorch, decoding, classic
core: yes

All functions take/return logits `(B, V)`.

* `apply_temperature(logits, T)` (T > 0).
* `top_k_filter(logits, k)` — set all but the k largest per row to `-inf`.
* `top_p_filter(logits, p)` — keep the smallest set of highest-probability tokens whose cumulative probability ≥ p;
  always keep the top token.
* `repetition_penalty(logits, prev_tokens, penalty)` — for tokens in `prev_tokens (B, L)`: divide positive logits by
  `penalty`, multiply negative ones by it.
* `sample(logits, T=1.0, k=None, p=None, generator=None)` → `(B,)` token ids; `T == 0` means greedy.
"""

import torch

def apply_temperature(logits, T):
    raise NotImplementedError  # TODO: implement


def top_k_filter(logits, k):
    raise NotImplementedError  # TODO: implement


def top_p_filter(logits, p):
    raise NotImplementedError  # TODO: implement


def repetition_penalty(logits, prev_tokens, penalty):
    raise NotImplementedError  # TODO: implement


def sample(logits, T=1.0, k=None, p=None, generator=None):
    raise NotImplementedError  # TODO: implement
