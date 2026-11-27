"""
title: Group anagrams
pattern: Arrays & hashing
level: medium
time: 20
frequency: very high
week: 2
day: 2
tags: hash-map, strings, canonical-form
complexity: O(n·k log k) with sorted keys, or O(n·k) with a 26-count key (k = word length)
hint: Two words are anagrams exactly when they map to the same canonical form.
hint: Use sorted(word) — or a tuple of letter counts — as a dict key.
follow-up: Words are very long and the alphabet is small: which key is cheaper, and why?
follow-up: Stream of words, return the size of the largest group seen so far after each word.

Group the words in `words` so that each group contains words that are anagrams of each other
(same letters with the same multiplicities; case-sensitive). Return a list of groups where

* words inside a group keep their **original relative order**, and
* groups are ordered by the position of their **first word** in the input.

```
group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
-> [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```
"""

def group_anagrams(words: list[str]) -> list[list[str]]:
    raise NotImplementedError  # TODO: implement
