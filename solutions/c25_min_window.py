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
    need: dict[str, int] = {}
    for ch in t:
        need[ch] = need.get(ch, 0) + 1
    missing = len(t)
    best = (float("inf"), 0)
    left = 0
    for right, ch in enumerate(s):
        if ch in need:
            if need[ch] > 0:
                missing -= 1
            need[ch] -= 1
        while missing == 0:
            if right - left + 1 < best[0]:
                best = (right - left + 1, left)
            lc = s[left]
            if lc in need:
                need[lc] += 1
                if need[lc] > 0:
                    missing += 1
            left += 1
    length, start = best
    return "" if length == float("inf") else s[start:start + length]
