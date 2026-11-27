"""
title: Edit distance with alignment — and word error rate for ASR
pattern: Dynamic programming
level: hard
time: 35
frequency: very high
week: 16
day: 4
tags: dp, strings, asr, wer, backtracking
complexity: O(n·m) time; O(n·m) space with backtrace (O(min(n, m)) for the distance only)
hint: D[i][j] = edit distance of a[:i] and b[:j]; it comes from D[i−1][j−1] (+0 match / +1 substitute), D[i−1][j] + 1 (delete), D[i][j−1] + 1 (insert).
hint: Keep the whole table, then walk back from (n, m) choosing a move that explains D[i][j] to recover the operations.
follow-up: Only the distance is needed and strings are long: keep two rows. Banded DP if you know distance ≤ k.
follow-up: Why is WER not bounded by 1? (insertions) — and why do ASR toolkits normalise text first?

Work on **sequences** (strings or lists of words).

* `edit_distance(a, b)` — Levenshtein distance (unit cost insert / delete / substitute);
* `align(a, b)` — `(distance, ops)` where `ops` is a list of `(op, x, y)` describing a minimal alignment from
  left to right: `("=", ai, bj)` match, `("S", ai, bj)` substitution, `("D", ai, None)` deletion from `a`,
  `("I", None, bj)` insertion of `b`'s element. When several alignments are minimal, prefer at each step back from
  the end: match/substitute, then delete, then insert.
* `wer(reference, hypothesis)` — word error rate: `(S + D + I) / len(reference words)` on whitespace-split
  strings, and also return the counts: `(wer, {"S": .., "D": .., "I": .., "N": ..})`. Empty reference → WER
  is 0.0 if the hypothesis is empty, else `float("inf")`.

```
edit_distance("kitten", "sitting") -> 3
wer("the cat sat on the mat", "the cat sit on mat") -> (0.333..., {"S": 1, "D": 1, "I": 0, "N": 6})
```
"""


def align(a, b):
    n, m_ = len(a), len(b)
    D = [[0] * (m_ + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m_ + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m_ + 1):
            sub = D[i - 1][j - 1] + (a[i - 1] != b[j - 1])
            D[i][j] = min(sub, D[i - 1][j] + 1, D[i][j - 1] + 1)
    ops = []
    i, j = n, m_
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("=" if a[i - 1] == b[j - 1] else "S", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            ops.append(("D", a[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, b[j - 1]))
            j -= 1
    return D[n][m_], ops[::-1]


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j - 1] + (a[i - 1] != b[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str):
    r, h = reference.split(), hypothesis.split()
    _, ops = align(r, h)
    counts = {k: sum(1 for o in ops if o[0] == k) for k in ("S", "D", "I")}
    counts["N"] = len(r)
    errs = counts["S"] + counts["D"] + counts["I"]
    if not r:
        return (0.0 if not h else float("inf")), counts
    return errs / len(r), counts
