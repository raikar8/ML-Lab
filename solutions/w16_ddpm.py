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
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    alphas = 1 - betas
    return {"betas": betas, "alphas": alphas, "alpha_bars": torch.cumprod(alphas, 0)}


def expand_like(v, x):
    return v.reshape(-1, *([1] * (x.ndim - 1))).to(x.dtype)


def q_sample(x0, t, noise, sched):
    ab = sched["alpha_bars"][t]
    return expand_like(ab.sqrt(), x0) * x0 + expand_like((1 - ab).sqrt(), x0) * noise


def ddpm_loss(model, x0, sched, generator=None):
    T = len(sched["betas"])
    t = torch.randint(0, T, (x0.shape[0],), generator=generator)
    noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
    return F.mse_loss(model(q_sample(x0, t, noise, sched), t), noise)


def p_sample_step(model, x_t, t, sched, noise):
    b, a, ab = (float(sched[k][t]) for k in ("betas", "alphas", "alpha_bars"))
    tt = torch.full((x_t.shape[0],), t, dtype=torch.long)
    eps = model(x_t, tt)
    mean = (x_t - b / (1 - ab) ** 0.5 * eps) / a ** 0.5
    return mean + (b ** 0.5) * noise if t > 0 else mean
