"""
title: Word break — can it be segmented, and all segmentations
pattern: Dynamic programming
level: medium
time: 30
frequency: very high
week: 19
day: 4
tags: dp, strings, memoisation, tokenisation
complexity: O(n · L) checks with a set and max word length L (O(n·L²) counting slicing)
hint: ok[i] = True if s[:i] can be segmented; ok[i] = any(ok[j] and s[j:i] in words) for j in [i − L, i).
hint: For all segmentations, memoise "all ways to segment s[i:]" — and only recurse where ok says it's possible.
follow-up: This is greedy/longest-match tokenisation's big brother: why do BPE/WordPiece not need DP at inference?
follow-up: Minimum number of words in a segmentation (shortest path in the DAG of positions).

`words` is a list of dictionary words.

* `can_break(s, words) -> bool` — can `s` be split into a sequence of dictionary words (reuse allowed)?
* `all_breaks(s, words)` — every segmentation as a space-joined string, sorted. Must be fast for strings that
  have **no** segmentation even when there are many partial matches (e.g. `"a" * 40 + "b"` with
  words `["a", "aa", "aaa"]`).

The empty string can be broken (into zero words): `all_breaks("", w) == [""]`.
"""

from functools import lru_cache

def can_break(s: str, words: list[str]) -> bool:
    raise NotImplementedError  # TODO: implement


def all_breaks(s: str, words: list[str]) -> list[str]:
    raise NotImplementedError  # TODO: implement
