"""
title: Debugging round — fix the broken training script
week: 27
day: 3
level: medium
time: 30
tags: pytorch, debugging, training
core: yes
stub: custom

The exercise file contains `train_classifier(X, y, n_classes, epochs, lr, seed)` and `evaluate(model, X, y)` with
**six planted bugs** (data, model, loss, optimiser and evaluation). The tests check that the model learns a simple task,
that evaluation doesn't change the model, and that inputs aren't mutated. Find and fix every bug, and for each one
write a one-line comment `# FIX: ...` explaining it. (This file is the fixed reference.)
"""
import torch
import torch.nn as nn


def make_model(d_in, n_classes):
    return nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes))


def train_classifier(X, y, n_classes, epochs=30, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    mean, std = X.mean(0), X.std(0) + 1e-8               # FIX: don't normalise X in place; keep stats for eval
    Xn = (X - mean) / std
    model = make_model(X.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)    # FIX: optimise the model's parameters
    loss_fn = nn.CrossEntropyLoss()                      # FIX: CE takes raw logits (no softmax first)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xn))
        for i in range(0, len(Xn), 32):
            idx = perm[i:i + 32]
            xb, yb = Xn[idx], y[idx]                      # FIX: same permutation for inputs and labels
            opt.zero_grad()                              # FIX: zero gradients every step
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.norm = (mean, std)
    return model


def evaluate(model, X, y):
    was = model.training
    model.eval()                                         # FIX: eval mode (dropout off), restore after
    with torch.no_grad():
        mean, std = model.norm
        acc = (model((X - mean) / std).argmax(-1) == y).float().mean().item()
    model.train(was)
    return acc
