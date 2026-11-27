"""
Serialize and deserialize a binary tree

`TreeNode` is given; values are integers (possibly negative). Implement `serialize(root) -> str` and
`deserialize(data) -> TreeNode | None` such that `deserialize(serialize(t))` reproduces the exact shape and
values. Must handle a 10 000-deep degenerate tree (no deep recursion). The string format is up to you, but
`serialize` must produce a `str` and must not use `pickle`, `json`, `repr`/`eval`.
"""

from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def serialize(root) -> str:
    raise NotImplementedError  # TODO: implement


def deserialize(data: str):
    raise NotImplementedError  # TODO: implement
