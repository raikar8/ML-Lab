"""
title: Decoding — temperature, top-k, top-p, repetition penalty, and a sampler
week: 27
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
    return logits / T


def top_k_filter(logits, k):
    kth = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    sl, si = logits.sort(dim=-1, descending=True)
    probs = sl.softmax(-1)
    remove = (probs.cumsum(-1) - probs) >= p
    remove[..., 0] = False
    sl = sl.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, si, sl)


def repetition_penalty(logits, prev_tokens, penalty):
    sel = logits.gather(-1, prev_tokens)
    sel = torch.where(sel > 0, sel / penalty, sel * penalty)
    return logits.scatter(-1, prev_tokens, sel)


def sample(logits, T=1.0, k=None, p=None, generator=None):
    if T == 0:
        return logits.argmax(-1)
    z = apply_temperature(logits, T)
    if k is not None:
        z = top_k_filter(z, k)
    if p is not None:
        z = top_p_filter(z, p)
    return torch.multinomial(z.softmax(-1), 1, generator=generator)[:, 0]
