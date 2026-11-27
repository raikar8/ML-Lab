"""
Trie with prefix counts and autocomplete

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
