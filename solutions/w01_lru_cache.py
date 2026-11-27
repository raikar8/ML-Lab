"""
title: LRU cache with O(1) get/put (no OrderedDict)
week: 1
day: 3
level: medium
time: 30
tags: python, data-structures, classic
core: yes

Implement `LRUCache(capacity)` with `get(key)` (returns the value or `-1`) and `put(key, value)`, both **O(1)**.
When inserting beyond capacity, evict the **least recently used** key. A `get` or `put` of an existing key
makes it most recently used. Also implement `__len__` and `keys()` returning keys from most to least recent.

Use a dict + a doubly linked list with sentinel head/tail nodes. Do not use `collections.OrderedDict`.
(The same structure is used for KV-cache block eviction and model caches.)
"""


class _Node:
    __slots__ = ("key", "val", "prev", "next")

    def __init__(self, key=None, val=None):
        self.key, self.val, self.prev, self.next = key, val, None, None


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map = {}
        self.head, self.tail = _Node(), _Node()  # head.next = most recent
        self.head.next, self.tail.prev = self.tail, self.head

    def unlink(self, n):
        n.prev.next, n.next.prev = n.next, n.prev

    def push_front(self, n):
        n.prev, n.next = self.head, self.head.next
        self.head.next.prev = n
        self.head.next = n

    def get(self, key):
        n = self.map.get(key)
        if n is None:
            return -1
        self.unlink(n)
        self.push_front(n)
        return n.val

    def put(self, key, value):
        n = self.map.get(key)
        if n is not None:
            n.val = value
            self.unlink(n)
            self.push_front(n)
            return
        if self.cap <= 0:
            return
        if len(self.map) >= self.cap:
            lru = self.tail.prev
            self.unlink(lru)
            del self.map[lru.key]
        n = _Node(key, value)
        self.map[key] = n
        self.push_front(n)

    def __len__(self):
        return len(self.map)

    def keys(self):
        out, n = [], self.head.next
        while n is not self.tail:
            out.append(n.key)
            n = n.next
        return out
