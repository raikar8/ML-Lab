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
        self.root = {"children": {}, "pass": 0, "end": 0}

    def insert(self, word: str) -> None:
        node = self.root
        node["pass"] += 1
        for ch in word:
            node = node["children"].setdefault(ch, {"children": {}, "pass": 0, "end": 0})
            node["pass"] += 1
        node["end"] += 1

    def find(self, prefix: str):
        node = self.root
        for ch in prefix:
            node = node["children"].get(ch)
            if node is None:
                return None
        return node

    def contains(self, word: str) -> bool:
        node = self.find(word)
        return node is not None and node["end"] > 0

    def remove(self, word: str) -> bool:
        if not self.contains(word):
            return False
        node = self.root
        node["pass"] -= 1
        for ch in word:
            child = node["children"][ch]
            child["pass"] -= 1
            if child["pass"] == 0:
                del node["children"][ch]   # prune the now-empty branch
                return True
            node = child
        node["end"] -= 1
        return True

    def count_prefix(self, prefix: str) -> int:
        node = self.find(prefix)
        return node["pass"] if node else 0

    def complete(self, prefix: str, k: int) -> list[str]:
        node = self.find(prefix)
        out: list[str] = []
        if node is None or k <= 0:
            return out
        stack = [(node, prefix)]
        while stack and len(out) < k:
            n, word = stack.pop()
            if n["end"] > 0:
                out.append(word)
            for ch in sorted(n["children"], reverse=True):   # reversed so the smallest is popped first
                stack.append((n["children"][ch], word + ch))
        return out
