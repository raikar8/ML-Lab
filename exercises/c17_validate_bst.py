"""
Validate a binary search tree, and find its k-th smallest value

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
