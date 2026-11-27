"""
title: A correct training loop — accumulation, clipping, eval mode, history
week: 9
day: 3
level: medium
time: 35
tags: pytorch, training, classic
core: yes

* `train_epoch(model, loader, opt, loss_fn, accum_steps=1, clip=None, scheduler=None)` — sets `model.train()`; for each
  batch `(x, y)` computes `loss_fn(model(x), y) / accum_steps`, backprops, and every `accum_steps` batches (and at the
  end of the epoch if there is a remainder) clips the global grad norm (if `clip`), steps the optimiser, steps the
  scheduler (if given), and zeroes grads. Return the list of **unscaled** per-batch losses (floats).
* `evaluate(model, loader, loss_fn)` → `(mean_loss, accuracy)` under `torch.inference_mode()` and `model.eval()`;
  restore the previous training mode afterwards. Accuracy = arg-max of outputs vs `y`.
"""

import torch

def train_epoch(model, loader, opt, loss_fn, accum_steps=1, clip=None, scheduler=None):
    raise NotImplementedError  # TODO: implement


def evaluate(model, loader, loss_fn):
    raise NotImplementedError  # TODO: implement
