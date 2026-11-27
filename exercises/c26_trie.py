"""
title: Trie with prefix counts and autocomplete
pattern: Trie
level: medium
time: 30
frequency: very high
week: 13
day: 4
tags: trie, strings, design
complexity: O(L) per insert/search/count for a word of length L; autocomplete O(L + size of the subtree visited)
hint: Each node: dict of children, a count of words passing through, and a count of words ending here.
hint: Autocomplete: walk to the prefix node, then DFS children in sorted order, collecting up to k words.
follow-up: Word Search II — find all dictionary words in a letter grid (trie + backtracking, prune exhausted branches).
follow-up: Memory: when would you use a double-array trie or a sorted array + binary search instead?

Implement `Trie` with

* `insert(word)` (duplicates allowed — they count twice), `remove(word) -> bool` (removes one copy; False if absent),
* `contains(word) -> bool`, `count_prefix(prefix) -> int` (number of stored words, with multiplicity, that start
  with `prefix`; the empty prefix counts everything),
* `complete(prefix, k)` — up to `k` **distinct** stored words starting with `prefix`, in lexicographic order.
"""

class Trie:
    def __init__(self):
        raise NotImplementedError  # TODO: implement

    def insert(self, word: str) -> None:
        raise NotImplementedError  # TODO: implement

    def find(self, prefix: str):
        raise NotImplementedError  # TODO: implement

    def contains(self, word: str) -> bool:
        raise NotImplementedError  # TODO: implement

    def remove(self, word: str) -> bool:
        raise NotImplementedError  # TODO: implement

    def count_prefix(self, prefix: str) -> int:
        raise NotImplementedError  # TODO: implement

    def complete(self, prefix: str, k: int) -> list[str]:
        raise NotImplementedError  # TODO: implement
