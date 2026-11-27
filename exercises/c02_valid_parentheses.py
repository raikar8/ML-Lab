"""
Balanced brackets — and where the first error is

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
