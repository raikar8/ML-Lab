"""
title: Export-ready streaming GRU model with explicit state (and torch.export)
week: 24
day: 4
level: medium
time: 35
tags: pytorch, deployment, streaming, audio
core: yes

`StreamingMaskNet(n_feat=40, hidden=64)` — per-frame mask estimator for enhancement/VAD:
`Linear(n_feat, hidden) → ReLU → GRU(hidden, hidden, batch_first) → Linear(hidden, n_feat) → sigmoid`.

* `forward(x, h)` — `x (B, T, F)`, `h (1, B, H)` → `(mask (B, T, F), h_new)`. State goes in and out explicitly — no
  hidden attributes — so the graph is exportable and the host owns the state.
* `init_state(batch)` → zeros `(1, B, H)`.
* `export_step(model, chunk_frames)` — `torch.export.export` the model on example inputs `(1, chunk_frames, F)` and
  `(1, 1, H)`; return the `ExportedProgram`.
Running chunk-by-chunk (carrying `h`) must equal running the whole sequence.
"""

import torch

import torch.nn as nn

class StreamingMaskNet(nn.Module):
    def __init__(self, n_feat=40, hidden=64):
        raise NotImplementedError  # TODO: implement

    def forward(self, x, h):
        raise NotImplementedError  # TODO: implement

    def init_state(self, batch):
        raise NotImplementedError  # TODO: implement


def export_step(model, chunk_frames):
    raise NotImplementedError  # TODO: implement
