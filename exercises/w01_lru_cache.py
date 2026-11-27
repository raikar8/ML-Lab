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
        raise NotImplementedError  # TODO: implement

    def unlink(self, n):
        raise NotImplementedError  # TODO: implement

    def push_front(self, n):
        raise NotImplementedError  # TODO: implement

    def get(self, key):
        raise NotImplementedError  # TODO: implement

    def put(self, key, value):
        raise NotImplementedError  # TODO: implement

    def __len__(self):
        raise NotImplementedError  # TODO: implement

    def keys(self):
        raise NotImplementedError  # TODO: implement
