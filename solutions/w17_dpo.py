"""
title: DPO loss from per-token log-probabilities
week: 17
day: 4
level: medium
time: 30
tags: pytorch, rlhf, dpo
core: yes

* `sequence_logprob(logits, labels, mask)` — `logits (B, T, V)`, `labels (B, T)`, `mask (B, T)` (1 on response tokens);
  return `(B,)` sums of log p(label) over masked positions.
* `dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1)` — inputs are `(B,)` sequence log-probs of chosen (w) and rejected (l)
  responses under the policy and the frozen reference. Return `(loss, chosen_reward, rejected_reward)` where
  `reward = beta · (log π − log π_ref)` (detached) and `loss = −mean(logsigmoid(r_w − r_l))` (with gradient).
"""
import torch
import torch.nn.functional as F


def sequence_logprob(logits, labels, mask):
    lp = torch.log_softmax(logits, -1).gather(-1, labels[..., None])[..., 0]
    return (lp * mask).sum(-1)


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    rw = beta * (pi_w - ref_w)
    rl = beta * (pi_l - ref_l)
    loss = -F.logsigmoid(rw - rl).mean()
    return loss, rw.detach(), rl.detach()
