"""
title: DDPM essentials — schedule, closed-form noising, loss, one reverse step
week: 16
day: 3
level: medium
time: 35
tags: pytorch, diffusion, generative
core: yes

* `linear_schedule(T, beta_start=1e-4, beta_end=0.02)` → dict of tensors `betas, alphas, alpha_bars` (length T).
* `q_sample(x0, t, noise, sched)` → `sqrt(ᾱ_t)·x0 + sqrt(1−ᾱ_t)·noise`; `t` is a LongTensor `(B,)`; broadcast over the
  remaining dims of `x0`.
* `ddpm_loss(model, x0, sched, generator=None)` — sample `t ~ U{0..T−1}` and `noise ~ N(0, I)`, return
  `mse(model(x_t, t), noise)`.
* `p_sample_step(model, x_t, t, sched, noise)` — one ancestral step from integer time `t` (same for the whole batch):
  `mean = (x_t − β_t/sqrt(1−ᾱ_t)·ε̂)/sqrt(α_t)`; add `sqrt(β_t)·noise` if `t > 0`.
"""

import torch

import torch.nn.functional as F

def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    raise NotImplementedError  # TODO: implement


def expand_like(v, x):
    raise NotImplementedError  # TODO: implement


def q_sample(x0, t, noise, sched):
    raise NotImplementedError  # TODO: implement


def ddpm_loss(model, x0, sched, generator=None):
    raise NotImplementedError  # TODO: implement


def p_sample_step(model, x_t, t, sched, noise):
    raise NotImplementedError  # TODO: implement
