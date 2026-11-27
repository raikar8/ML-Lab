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
    n, m_ = len(a), len(b)
    L = [[0] * (m_ + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):        # suffix table: L[i][j] = LCS of a[i:], b[j:]
        for j in range(m_ - 1, -1, -1):
            L[i][j] = L[i + 1][j + 1] + 1 if a[i] == b[j] else max(L[i + 1][j], L[i][j + 1])
    return L


def lcs_length(a, b) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def diff(a, b):
    L = table(a, b)
    i = j = 0
    out = []
    while i < len(a) or j < len(b):
        if i < len(a) and j < len(b) and a[i] == b[j]:
            out.append((" ", a[i])); i += 1; j += 1
        elif i < len(a) and (j == len(b) or L[i + 1][j] >= L[i][j + 1]):
            out.append(("-", a[i])); i += 1
        else:
            out.append(("+", b[j])); j += 1
    return out


def lcs(a, b) -> list:
    return [x for tag, x in diff(a, b) if tag == " "]
