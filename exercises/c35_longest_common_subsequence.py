"""
title: Longest common subsequence — length, one LCS, and a diff
pattern: Dynamic programming
level: medium
time: 30
frequency: very high
week: 18
day: 2
tags: dp, strings, diff
complexity: O(n·m) time and space (O(min(n, m)) space for the length only)
hint: L[i][j] = L[i−1][j−1] + 1 if a[i−1] == b[j−1], else max(L[i−1][j], L[i][j−1]).
hint: Rebuild by walking back from (n, m): on a match take the character diagonally; otherwise move toward the larger neighbour.
follow-up: That diff is what `git diff` / `difflib` do (Myers' O(ND) algorithm is faster when the files are similar).
follow-up: Longest common substring (contiguous) — what changes in the recurrence?

* `lcs_length(a, b)` — length of the longest common subsequence of sequences `a`, `b`;
* `lcs(a, b)` — one LCS as a list of elements;
* `diff(a, b)` — list of `(" ", x)`, `("-", x)`, `("+", y)` lines turning `a` into `b`, keeping an LCS as the
  unchanged lines (`" "`). Prefer `"-"` before `"+"` when both are possible at the same point.

```
lcs_length("abcde", "ace") -> 3
diff(["a", "b", "c"], ["a", "c", "d"]) -> [(" ", "a"), ("-", "b"), (" ", "c"), ("+", "d")]
```
"""

def table(a, b):
    raise NotImplementedError  # TODO: implement


def lcs_length(a, b) -> int:
    raise NotImplementedError  # TODO: implement


def diff(a, b):
    raise NotImplementedError  # TODO: implement


def lcs(a, b) -> list:
    raise NotImplementedError  # TODO: implement
