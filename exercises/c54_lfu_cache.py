"""
LFU cache with O(1) operations

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
