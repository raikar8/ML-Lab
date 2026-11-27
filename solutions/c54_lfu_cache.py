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
        self.cap = capacity
        self.val: dict = {}
        self.freq: dict = {}
        self.buckets: dict[int, dict] = {}   # freq -> ordered "set" of keys (dict with None values)
        self.min_freq = 0

    def touch(self, key) -> None:
        f = self.freq[key]
        del self.buckets[f][key]
        if not self.buckets[f]:
            del self.buckets[f]
            if self.min_freq == f:
                self.min_freq = f + 1
        self.freq[key] = f + 1
        self.buckets.setdefault(f + 1, {})[key] = None

    def get(self, key):
        if key not in self.val:
            return -1
        self.touch(key)
        return self.val[key]

    def put(self, key, value) -> None:
        if self.cap <= 0:
            return
        if key in self.val:
            self.val[key] = value
            self.touch(key)
            return
        if len(self.val) >= self.cap:
            bucket = self.buckets[self.min_freq]
            old = next(iter(bucket))
            del bucket[old]
            if not bucket:
                del self.buckets[self.min_freq]
            del self.val[old], self.freq[old]
        self.val[key], self.freq[key] = value, 1
        self.buckets.setdefault(1, {})[key] = None
        self.min_freq = 1

    def __len__(self) -> int:
        return len(self.val)
