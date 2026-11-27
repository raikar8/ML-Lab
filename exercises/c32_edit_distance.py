"""
Edit distance with alignment — and word error rate for ASR

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
    raise NotImplementedError  # TODO: implement


def edit_distance(a, b) -> int:
    raise NotImplementedError  # TODO: implement


def wer(reference: str, hypothesis: str):
    raise NotImplementedError  # TODO: implement
