"""
title: CTC loss via the forward algorithm, and greedy decoding
week: 8
day: 3
level: hard
time: 50
tags: numpy, ctc, asr, probability
core: yes

* `ctc_loss(log_probs, target, blank=0)` — `log_probs` is a NumPy array `(T, V)` of per-frame log-probabilities, `target`
  a list of label ids (no blanks). Return the negative log-likelihood `−log p(target | x)` summed over all valid
  alignments, computed with the **forward algorithm in log space** over the extended sequence
  `[blank, y1, blank, y2, …, yU, blank]`. Allowed transitions: stay, advance by 1, or skip a blank when the two labels
  differ. Return `inf` if no alignment exists.
* `greedy_decode(log_probs, blank=0)` — arg-max per frame, merge repeats, remove blanks → list of ids.
"""

import numpy as np

def logadd(*xs):
    raise NotImplementedError  # TODO: implement


def ctc_loss(log_probs, target, blank=0):
    raise NotImplementedError  # TODO: implement


def greedy_decode(log_probs, blank=0):
    raise NotImplementedError  # TODO: implement
