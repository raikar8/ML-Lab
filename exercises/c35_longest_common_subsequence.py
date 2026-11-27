"""
Longest common subsequence — length, one LCS, and a diff

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
