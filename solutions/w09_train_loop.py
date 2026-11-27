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
    model.train()
    losses = []
    opt.zero_grad(set_to_none=True)

    def do_step():
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        opt.zero_grad(set_to_none=True)

    i = -1
    for i, (x, y) in enumerate(loader):
        loss = loss_fn(model(x), y)
        (loss / accum_steps).backward()
        losses.append(loss.item())
        if (i + 1) % accum_steps == 0:
            do_step()
    if (i + 1) % accum_steps != 0:
        do_step()
    return losses


def evaluate(model, loader, loss_fn):
    was = model.training
    model.eval()
    tot, correct, n = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            out = model(x)
            tot += loss_fn(out, y).item() * len(y)
            correct += (out.argmax(-1) == y).sum().item()
            n += len(y)
    model.train(was)
    return tot / n, correct / n
