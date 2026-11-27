"""
title: Padding collate, masked mean, and a length-bucketing batch sampler
week: 9
day: 1
level: medium
time: 35
tags: pytorch, data
core: yes

* `pad_collate(batch)` — `batch` is a list of `(features (T_i, F) float tensor, label int)`. Return a dict with
  `x (B, T_max, F)` zero-padded, `lengths (B,)` long, `mask (B, T_max)` bool (True = real frame), `y (B,)` long.
* `masked_mean(x, mask)` — mean over time of `(B, T, F)` using only real frames → `(B, F)`.
* `BucketBatchSampler(lengths, batch_size, seed=0, bucket_size=None)` — an iterable of index lists for `DataLoader(
  batch_sampler=...)`: shuffle indices, split them into chunks of `bucket_size` (default `batch_size * 20`), sort each
  chunk by length, cut it into batches, then shuffle the batch order. Every index appears exactly once per epoch;
  `set_epoch(e)` changes the shuffle; `__len__` = number of batches.
"""
import random

import torch


def pad_collate(batch):
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    x = torch.nn.utils.rnn.pad_sequence(list(feats), batch_first=True)
    mask = torch.arange(x.shape[1])[None] < lengths[:, None]
    return {"x": x, "lengths": lengths, "mask": mask, "y": torch.tensor(labels, dtype=torch.long)}


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1)


class BucketBatchSampler:
    def __init__(self, lengths, batch_size, seed=0, bucket_size=None):
        self.lengths, self.bs, self.seed = list(lengths), batch_size, seed
        self.bucket = bucket_size or batch_size * 20
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def batches(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        out = []
        for i in range(0, len(idx), self.bucket):
            chunk = sorted(idx[i:i + self.bucket], key=lambda j: self.lengths[j])
            out += [chunk[k:k + self.bs] for k in range(0, len(chunk), self.bs)]
        rng.shuffle(out)
        return out

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return len(self.batches())
