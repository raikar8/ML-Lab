"""
title: Validate a binary search tree, and find its k-th smallest value
pattern: Trees
level: medium
time: 25
frequency: very high
week: 9
day: 2
tags: tree, bst, inorder, invariants
complexity: O(n) time for validation; O(h + k) for k-th smallest; O(h) stack space
hint: Checking only parent/child pairs is wrong — every node must lie in an interval inherited from all ancestors.
hint: Equivalently, an inorder traversal of a BST is strictly increasing; stop early after k nodes for the k-th.
follow-up: The tree is modified often and k-th smallest is asked often: store subtree sizes in each node.
follow-up: Two nodes of a BST were swapped by mistake — restore it in O(1) extra space (Morris traversal).

`TreeNode` and `from_level` are given. A valid BST has **strictly** smaller values in every left subtree and
strictly larger values in every right subtree.

* `is_bst(root) -> bool` (an empty tree is valid);
* `kth_smallest(root, k)` — k-th smallest value (1-indexed) of a valid BST; raise `ValueError` if `k` is out
  of range. Use an iterative inorder traversal that stops after `k` nodes.

```
is_bst(from_level([5, 1, 7, None, None, 6, 8])) -> True
is_bst(from_level([5, 1, 7, None, None, 4, 8])) -> False   # 4 is in 5's right subtree
```
"""

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

def is_bst(root) -> bool:
    raise NotImplementedError  # TODO: implement


def kth_smallest(root, k: int) -> int:
    raise NotImplementedError  # TODO: implement
