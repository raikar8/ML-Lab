"""
title: Export-ready streaming GRU model with explicit state (and torch.export)
week: 25
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
        super().__init__()
        self.inp = nn.Linear(n_feat, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)
        self.hidden = hidden

    def forward(self, x, h):
        z, h_new = self.gru(torch.relu(self.inp(x)), h)
        return torch.sigmoid(self.out(z)), h_new

    def init_state(self, batch):
        return torch.zeros(1, batch, self.hidden)


def export_step(model, chunk_frames):
    x = torch.zeros(1, chunk_frames, model.inp.in_features)
    return torch.export.export(model.eval(), (x, model.init_state(1)))
