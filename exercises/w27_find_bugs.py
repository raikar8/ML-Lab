"""
title: Debugging round — fix the broken training script
week: 27
day: 3
level: medium
time: 30
tags: pytorch, debugging, training
core: yes
stub: custom

This training script runs but doesn't work. There are **six planted bugs** (data, model, loss, optimiser and
evaluation). The tests check that the model learns a simple task, that evaluation doesn't change the model, and that
the caller's inputs aren't mutated. Find and fix every bug; add a `# FIX: ...` comment for each.

Method: check the loss at init (≈ ln K), try to overfit one batch, then walk data → model → loss → optimiser → eval.
"""
import torch
import torch.nn as nn


def make_model(d_in, n_classes):
    return nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes))


def train_classifier(X, y, n_classes, epochs=30, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    mean, std = X.mean(0), X.std(0) + 1e-8
    X -= mean
    X /= std
    model = make_model(X.shape[1], n_classes)
    opt = torch.optim.Adam(make_model(X.shape[1], n_classes).parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            xb = X[perm[i:i + 32]]
            yb = y[i:i + 32]
            loss = loss_fn(torch.softmax(model(xb), -1), yb)
            loss.backward()
            opt.step()
    model.norm = (mean, std)
    return model


def evaluate(model, X, y):
    mean, std = model.norm
    acc = (model((X - mean) / std).argmax(-1) == y).float().mean().item()
    return acc
