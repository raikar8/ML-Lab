"""
title: Minimum window substring
pattern: Sliding window
level: hard
time: 35
frequency: very high
week: 13
day: 2
tags: sliding-window, hash-map, two-pointers
complexity: O(|s| + |t|) time, O(alphabet) space
hint: Expand the right edge until the window covers t (track how many required characters are still missing).
hint: Then shrink from the left while it still covers t, recording the best window; one missing char → expand again.
follow-up: Characters of t must appear in order (minimum window subsequence) — DP or two-pass scanning.
follow-up: Smallest range covering at least one element from each of k sorted lists (heap + window).

Return the shortest substring of `s` that contains every character of `t` **with multiplicity**
(if `t = "aab"` the window needs two `a`s and one `b`). Ties → leftmost. If none exists, return `""`.
`t` is non-empty. Linear time.

```
min_window("ADOBECODEBANC", "ABC") -> "BANC"
min_window("a", "aa")              -> ""
```
"""

def min_window(s: str, t: str) -> str:
    raise NotImplementedError  # TODO: implement
