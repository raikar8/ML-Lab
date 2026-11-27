"""
title: LFU cache with O(1) operations
pattern: Design
level: hard
time: 40
frequency: medium
week: 27
day: 4
tags: design, hash-map, linked-hash, cache
complexity: O(1) per get/put
hint: Map key → (value, freq), and freq → an insertion-ordered set of keys (a dict works: it keeps order). Track min_freq.
hint: On access, move the key from bucket f to f+1; if bucket f empties and f was min_freq, min_freq += 1. A new key resets min_freq to 1.
follow-up: Why do real caches (CPU, CDN, KV-cache eviction) rarely use pure LFU? (frequency never ages → use decay, TinyLFU, ARC)
follow-up: Make it thread-safe; what's the contention point?

`LFUCache(capacity)` with `get(key)` (value or `-1`) and `put(key, value)`, both O(1). When full, evict the key
with the **lowest access count**; among those, the **least recently used**. Both `get` and `put` on an existing
key count as an access. Capacity 0 stores nothing. Also implement `__len__`.

(Don't use a heap — that's O(log n).)
"""

class LFUCache:
    def __init__(self, capacity: int):
        raise NotImplementedError  # TODO: implement

    def touch(self, key) -> None:
        raise NotImplementedError  # TODO: implement

    def get(self, key):
        raise NotImplementedError  # TODO: implement

    def put(self, key, value) -> None:
        raise NotImplementedError  # TODO: implement

    def __len__(self) -> int:
        raise NotImplementedError  # TODO: implement
