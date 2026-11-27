"""
title: Auditing an LLM judge — position bias, debiased win rate, Cohen's kappa
week: 20
day: 5
level: easy
time: 25
tags: python, evaluation, llm-judge
core: no

Each pairwise comparison is judged twice: once as (A, B) and once swapped (B, A). A verdict is `"first"`, `"second"` or
`"tie"`.

* `position_bias(verdicts_ab, verdicts_ba)` — fraction of items where the judge picked the **same position** in both orders
  (both "first" or both "second") — inconsistent, position-driven judgments.
* `debiased_win_rate(verdicts_ab, verdicts_ba)` — model A's win rate counting A-wins from both orders: in (A,B) "first" = A
  wins, in (B,A) "second" = A wins; ties count 0.5; average over the 2·n judgments.
* `cohens_kappa(a, b)` — agreement between two label lists beyond chance: `(p_o − p_e) / (1 − p_e)`.
"""
from collections import Counter


def position_bias(verdicts_ab, verdicts_ba):
    same = sum(1 for x, y in zip(verdicts_ab, verdicts_ba) if x == y and x in ("first", "second"))
    return same / len(verdicts_ab)


def debiased_win_rate(verdicts_ab, verdicts_ba):
    score = 0.0
    for x in verdicts_ab:
        score += 1 if x == "first" else 0.5 if x == "tie" else 0
    for y in verdicts_ba:
        score += 1 if y == "second" else 0.5 if y == "tie" else 0
    return score / (len(verdicts_ab) + len(verdicts_ba))


def cohens_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
