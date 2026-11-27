"""
title: GRPO — group-relative advantages, clipped objective, k3 KL
week: 17
day: 5
level: hard
time: 45
tags: pytorch, rl, grpo, chinese-models
core: yes

For one prompt with a group of G sampled responses:

* `group_advantages(rewards, eps=1e-6, scale=True)` — `(G,)` → `(r − mean) / (std + eps)` (population std, `unbiased=False`);
  with `scale=False` (Dr. GRPO) only subtract the mean.
* `grpo_loss(logp_new, logp_old, logp_ref, adv, mask, clip_eps=0.2, beta=0.04, clip_high=None)` — token log-probs
  `(G, T)` under the current, rollout and reference policies, `adv (G,)`, `mask (G, T)` response mask.
  Per token: `ratio = exp(new − old)`, surrogate `min(ratio·A, clip(ratio, 1−ε, 1+ε_high)·A)` with `ε_high = clip_high or ε`
  (DAPO's clip-higher), KL estimator `k3 = exp(ref − new) − (ref − new) − 1`. Per-token objective
  `surrogate − beta·k3`; average over each response's tokens, then over the group; return the **negative** (a loss).
"""
import torch


def group_advantages(rewards, eps=1e-6, scale=True):
    a = rewards - rewards.mean()
    return a / (rewards.std(unbiased=False) + eps) if scale else a


def grpo_loss(logp_new, logp_old, logp_ref, adv, mask, clip_eps=0.2, beta=0.04, clip_high=None):
    hi = clip_high if clip_high is not None else clip_eps
    ratio = torch.exp(logp_new - logp_old)
    A = adv[:, None]
    surr = torch.minimum(ratio * A, torch.clamp(ratio, 1 - clip_eps, 1 + hi) * A)
    d = logp_ref - logp_new
    k3 = torch.exp(d) - d - 1
    per_tok = (surr - beta * k3) * mask
    per_seq = per_tok.sum(1) / mask.sum(1).clamp_min(1)
    return -per_seq.mean()
