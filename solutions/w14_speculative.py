"""
title: Speculative sampling — lossless draft-and-verify
week: 14
day: 5
level: hard
time: 45
tags: pytorch, speculative, inference, decoding
core: yes

Work with explicit probability vectors so the maths is testable.

* `accept_or_resample(p, q, x, u)` — target probs `p (V,)`, draft probs `q (V,)`, drafted token `x`, uniform `u ∈ [0,1)`.
  Accept if `u < min(1, p[x]/q[x])` → return `(True, x)`; otherwise return `(False, y)` where `y` is drawn from the
  residual `norm(max(0, p − q))` using `torch.multinomial` with the global RNG.
* `speculative_step(target_probs, draft_probs, draft_tokens, generator=None)` — `draft_tokens` `(γ,)`;
  `draft_probs (γ, V)`; `target_probs (γ+1, V)` (the target's distribution at each drafted position plus one bonus).
  Walk the draft tokens in order, stop at the first rejection (emit the resampled token); if all are accepted, also
  sample one bonus token from `target_probs[γ]`. Return the list of emitted token ids (length 1…γ+1).
* `expected_tokens(alpha, gamma)` → `(1 − α^(γ+1)) / (1 − α)`.
"""
import torch


def accept_or_resample(p, q, x, u):
    if u < min(1.0, float(p[x] / q[x])):
        return True, int(x)
    r = torch.clamp(p - q, min=0)
    r = r / r.sum()
    return False, int(torch.multinomial(r, 1))


def speculative_step(target_probs, draft_probs, draft_tokens, generator=None):
    out = []
    for i, x in enumerate(draft_tokens.tolist()):
        u = float(torch.rand((), generator=generator))
        ok, tok = accept_or_resample(target_probs[i], draft_probs[i], x, u)
        out.append(tok)
        if not ok:
            return out
    out.append(int(torch.multinomial(target_probs[len(draft_tokens)], 1, generator=generator)))
    return out


def expected_tokens(alpha, gamma):
    if alpha == 1:
        return gamma + 1
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)
