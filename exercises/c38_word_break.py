"""
Word break — can it be segmented, and all segmentations

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
