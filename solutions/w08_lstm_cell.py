"""
title: LSTM cell and sequence unroll matching nn.LSTM
week: 8
day: 1
level: medium
time: 30
tags: pytorch, rnn, audio
core: no

* `lstm_cell(x, h, c, W_ih, W_hh, b_ih, b_hh)` — one step for a batch: `x (B, D)`, `h, c (B, H)`; weights in PyTorch's
  layout `W_ih (4H, D)`, `W_hh (4H, H)` with gate order **i, f, g, o**. Return `(h_new, c_new)`.
* `lstm_forward(x, lstm)` — run over `x (B, T, D)` using the parameters of a single-layer `nn.LSTM(batch_first=True)`
  (`lstm.weight_ih_l0` etc.), starting from zero state; return `(outputs (B, T, H), (h_T, c_T))`.
"""
import torch


def lstm_cell(x, h, c, W_ih, W_hh, b_ih, b_hh):
    gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh
    i, f, g, o = gates.chunk(4, dim=-1)
    i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
    c = f * c + i * g
    return o * torch.tanh(c), c


def lstm_forward(x, lstm):
    B, T, _ = x.shape
    H = lstm.hidden_size
    h = x.new_zeros(B, H)
    c = x.new_zeros(B, H)
    outs = []
    for t in range(T):
        h, c = lstm_cell(x[:, t], h, c, lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0)
        outs.append(h)
    return torch.stack(outs, 1), (h, c)
