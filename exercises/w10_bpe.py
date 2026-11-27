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
    raise NotImplementedError  # TODO: implement


def train_bpe(text, num_merges):
    raise NotImplementedError  # TODO: implement


def encode(text, merges):
    raise NotImplementedError  # TODO: implement


def decode(symbols):
    raise NotImplementedError  # TODO: implement
