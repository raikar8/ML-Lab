"""
title: Balanced brackets — and where the first error is
pattern: Stack
level: easy
time: 15
frequency: very high
week: 1
day: 4
tags: stack, parsing
complexity: O(n) time, O(n) space
hint: The most recent unmatched opener is the only one a closer may match — that's a stack.
hint: Two failure modes: a closer that doesn't match the top (or an empty stack), and openers left over at the end.
follow-up: Only one bracket type: do it in O(1) space with a counter.
follow-up: Return the minimum number of insertions to make the string balanced.

`first_error(s)` checks a string containing brackets `()[]{}` and any other characters (which are ignored).
Return `-1` if every bracket is matched and properly nested. Otherwise return the **index of the first
character that proves the string is invalid**: a closer with no matching opener, or — if the string ends with
unclosed openers — the index of the **earliest unclosed opener**.

Also write `is_balanced(s) -> bool`.

```
first_error("a(b[c]{d}e)f") -> -1
first_error("(]")           -> 1
first_error("([)]")         -> 2
first_error("x((y)")        -> 1      # the opener at 1 is never closed
```
"""

PAIRS = {")": "(", "]": "[", "}": "{"}

def first_error(s: str) -> int:
    raise NotImplementedError  # TODO: implement


def is_balanced(s: str) -> bool:
    raise NotImplementedError  # TODO: implement
