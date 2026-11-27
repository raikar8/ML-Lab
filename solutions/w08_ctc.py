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
    m = max(xs)
    if m == -np.inf:
        return -np.inf
    return m + np.log(sum(np.exp(x - m) for x in xs))


def ctc_loss(log_probs, target, blank=0):
    T = log_probs.shape[0]
    ext = [blank]
    for y in target:
        ext += [y, blank]
    S = len(ext)
    alpha = np.full(S, -np.inf)
    alpha[0] = log_probs[0, blank]
    if S > 1:
        alpha[1] = log_probs[0, ext[1]]
    for t in range(1, T):
        new = np.full(S, -np.inf)
        for s in range(S):
            terms = [alpha[s]]
            if s >= 1:
                terms.append(alpha[s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                terms.append(alpha[s - 2])
            new[s] = logadd(*terms) + log_probs[t, ext[s]]
        alpha = new
    ll = logadd(alpha[S - 1], alpha[S - 2]) if S > 1 else alpha[0]
    return float(-ll)


def greedy_decode(log_probs, blank=0):
    best = log_probs.argmax(1)
    out, prev = [], None
    for b in best:
        if b != prev and b != blank:
            out.append(int(b))
        prev = b
    return out
