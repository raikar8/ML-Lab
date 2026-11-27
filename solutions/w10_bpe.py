"""
title: Train and apply a byte-pair-encoding tokenizer
week: 10
day: 1
level: medium
time: 45
tags: python, tokenisation, transformers
core: no

Character-level BPE (no regex pre-tokenisation needed; treat the text as one sequence of characters).

* `train_bpe(text, num_merges)` → list of merges `[(a, b), …]` in learned order: repeatedly count adjacent symbol
  pairs in the current sequence, merge the most frequent pair (ties: lexicographically smallest pair) into the symbol
  `a + b` everywhere (left to right, non-overlapping), stop early when no pair occurs twice.
* `encode(text, merges)` → list of symbols (strings) after applying the merges **in order**.
* `decode(symbols)` → the original text.
"""
from collections import Counter


def merge_pair(seq, pair):
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
            out.append(seq[i] + seq[i + 1])
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_bpe(text, num_merges):
    seq = list(text)
    merges = []
    for _ in range(num_merges):
        counts = Counter(zip(seq, seq[1:]))
        if not counts:
            break
        best = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if best[1] < 2:
            break
        merges.append(best[0])
        seq = merge_pair(seq, best[0])
    return merges


def encode(text, merges):
    seq = list(text)
    for p in merges:
        seq = merge_pair(seq, p)
    return seq


def decode(symbols):
    return "".join(symbols)
