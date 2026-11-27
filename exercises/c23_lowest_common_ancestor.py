"""
Lowest common ancestor — binary tree, BST, and with parent pointers

`TreeNode` and `from_level` are given; nodes are compared by **identity**, and `p`, `q` are guaranteed to be in
the tree. Implement

* `lca(root, p, q)` — for any binary tree, iterative or recursive (trees up to 3000 deep must work → avoid deep
  recursion, e.g. build a parent map with a stack);
* `lca_bst(root, p, q)` — for a BST, O(h) without visiting other branches.

Return the LCA **node**. A node is its own ancestor.
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

def lca(root, p, q):
    raise NotImplementedError  # TODO: implement


def lca_bst(root, p, q):
    raise NotImplementedError  # TODO: implement
