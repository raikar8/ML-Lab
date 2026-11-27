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
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def flatten(nested):
    stack = [iter(nested)]
    while stack:
        for x in stack[-1]:
            if isinstance(x, list):
                stack.append(iter(x))
                break
            yield x
        else:
            stack.pop()


def group_by(items, key):
    out = defaultdict(list)
    for x in items:
        out[key(x)].append(x)
    return dict(out)


def rle_encode(s: str):
    out = []
    for ch in s:
        if out and out[-1][0] == ch:
            out[-1] = (ch, out[-1][1] + 1)
        else:
            out.append((ch, 1))
    return out


def rle_decode(pairs) -> str:
    return "".join(ch * n for ch, n in pairs)


def rank_players(rows):
    return [name for name, _, _ in sorted(rows, key=lambda r: (-r[1], r[2], r[0]))]


def make_grid(r: int, c: int, fill=0):
    return [[fill] * c for _ in range(r)]
