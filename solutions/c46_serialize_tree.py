"""
title: Serialize and deserialize a binary tree
pattern: Trees
level: hard
time: 30
frequency: very high
week: 23
day: 4
tags: tree, serialization, bfs, design
complexity: O(n) time and space for both directions
hint: Pre-order with an explicit marker for null children is enough to rebuild the shape uniquely.
hint: Use a token separator and handle negative / multi-digit values; avoid recursion depth problems with an explicit stack or BFS.
follow-up: Serialize a BST more compactly (pre-order without null markers — the BST property restores the shape).
follow-up: Versioning a model-config tree on disk: why would you prefer a schema'd format (JSON/protobuf) here?

`TreeNode` is given; values are integers (possibly negative). Implement `serialize(root) -> str` and
`deserialize(data) -> TreeNode | None` such that `deserialize(serialize(t))` reproduces the exact shape and
values. Must handle a 10 000-deep degenerate tree (no deep recursion). The string format is up to you, but
`serialize` must produce a `str` and must not use `pickle`, `json`, `repr`/`eval`.
"""
from collections import deque


# given
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def serialize(root) -> str:
    out = []
    q = deque([root])
    while q:
        node = q.popleft()
        if node is None:
            out.append("#")
        else:
            out.append(str(node.val))
            q.append(node.left)
            q.append(node.right)
    while out and out[-1] == "#":
        out.pop()
    return ",".join(out)


def deserialize(data: str):
    if not data:
        return None
    toks = data.split(",")
    root = TreeNode(int(toks[0]))
    q = deque([root])
    i = 1
    while q and i < len(toks):
        node = q.popleft()
        for side in ("left", "right"):
            if i < len(toks) and toks[i] != "#":
                child = TreeNode(int(toks[i]))
                setattr(node, side, child)
                q.append(child)
            i += 1
    return root
