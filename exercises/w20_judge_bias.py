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
    raise NotImplementedError  # TODO: implement


def debiased_win_rate(verdicts_ab, verdicts_ba):
    raise NotImplementedError  # TODO: implement


def cohens_kappa(a, b):
    raise NotImplementedError  # TODO: implement
