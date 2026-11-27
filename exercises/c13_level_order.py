"""
title: Binary tree level-order traversal — levels, zigzag and right-side view
pattern: Trees
level: medium
time: 25
frequency: very high
week: 7
day: 2
tags: tree, bfs, queue
complexity: O(n) time, O(width) space
hint: BFS with a queue; process exactly len(queue) nodes per round to know where a level ends.
hint: Zigzag = reverse every other level; right-side view = last node of each level.
follow-up: Do it with DFS carrying the depth — when is DFS preferable?
follow-up: Vertical-order traversal (group by column, then row, then value).

`TreeNode` and `from_level` are given. Implement

* `level_order(root)` — list of levels, each a list of values left to right;
* `zigzag(root)` — like level_order but alternate direction, starting left→right;
* `right_view(root)` — the value of the rightmost node on each level.

An empty tree gives `[]` for all three.

```
root = from_level([3, 9, 20, None, None, 15, 7])
level_order(root) -> [[3], [9, 20], [15, 7]]
zigzag(root)      -> [[3], [20, 9], [15, 7]]
right_view(root)  -> [3, 20, 7]
```
"""

from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def from_level(vals):
    """Build a tree from level-order values with None for missing children: [3, 9, 20, None, None, 15, 7]."""
    if not vals or vals[0] is None:
        return None
    it = iter(vals)
    root = TreeNode(next(it))
    queue = [root]
    for node in queue:
        for side in ("left", "right"):
            v = next(it, None)
            if v is not None:
                child = TreeNode(v)
                setattr(node, side, child)
                queue.append(child)
    return root

def level_order(root) -> list[list[int]]:
    raise NotImplementedError  # TODO: implement


def zigzag(root) -> list[list[int]]:
    raise NotImplementedError  # TODO: implement


def right_view(root) -> list[int]:
    raise NotImplementedError  # TODO: implement
