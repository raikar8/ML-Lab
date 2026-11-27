"""
title: Everyday Python for interviews — dedupe, flatten, group, run-length coding, sorting with keys
week: 0
day: 4
level: easy
time: 30
tags: python, data-structures, foundations
core: yes

Pure Python (standard library allowed):

* `dedupe(items)` — remove duplicates keeping the **first** occurrence order; works for hashable items in O(n).
* `flatten(nested)` — generator yielding the non-list leaves of arbitrarily nested **lists** (tuples and strings are
  leaves), left to right; must handle 10 000 levels of nesting (no recursion).
* `group_by(items, key)` — dict `key(x) -> list of x` in input order (`defaultdict` fine).
* `rle_encode(s)` / `rle_decode(pairs)` — run-length coding: `"aaabcc" -> [("a", 3), ("b", 1), ("c", 2)]` and back.
* `rank_players(rows)` — `rows` are `(name, score, time)`; return names sorted by score **descending**, then time
  **ascending**, then name ascending (one `sorted` call with a key).
* `make_grid(r, c, fill)` — an r×c list of lists whose rows are **independent** objects.
"""

from collections import defaultdict

def dedupe(items):
    raise NotImplementedError  # TODO: implement


def flatten(nested):
    raise NotImplementedError  # TODO: implement


def group_by(items, key):
    raise NotImplementedError  # TODO: implement


def rle_encode(s: str):
    raise NotImplementedError  # TODO: implement


def rle_decode(pairs) -> str:
    raise NotImplementedError  # TODO: implement


def rank_players(rows):
    raise NotImplementedError  # TODO: implement


def make_grid(r: int, c: int, fill=0):
    raise NotImplementedError  # TODO: implement
