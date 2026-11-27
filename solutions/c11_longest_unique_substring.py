"""
title: Longest substring without repeating characters
pattern: Sliding window
level: medium
time: 20
frequency: very high
week: 6
day: 2
tags: sliding-window, hash-map, strings
complexity: O(n) time, O(alphabet) space
hint: Grow a window to the right; when the new character is already inside, move the left edge past its last occurrence.
hint: Store last_seen[ch]; left = max(left, last_seen[ch] + 1) — the max matters.
follow-up: Longest substring with at most k distinct characters.
follow-up: Return all maximal windows, not just the length.

Return `(length, start)` of the longest substring of `s` with all characters distinct. Ties go to the
**earliest start**. For an empty string return `(0, 0)`.

```
longest_unique("abcabcbb") -> (3, 0)   # "abc"
longest_unique("pwwkew")   -> (3, 2)   # "wke"
```
"""


def longest_unique(s: str) -> tuple[int, int]:
    last: dict[str, int] = {}
    left = best_len = best_start = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1
        last[ch] = right
        if right - left + 1 > best_len:
            best_len, best_start = right - left + 1, left
    return best_len, best_start
