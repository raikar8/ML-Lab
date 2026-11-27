"""
title: Lowest common ancestor — binary tree, BST, and with parent pointers
pattern: Trees
level: medium
time: 25
frequency: very high
week: 12
day: 2
tags: tree, recursion, bst
complexity: O(n) for a general tree, O(h) for a BST, O(depth) with parent pointers
hint: General tree: a node is the LCA if p and q are found in different subtrees, or it is one of them and the other is below.
hint: BST: if both values are smaller go left, both larger go right, otherwise you're standing on the LCA.
follow-up: Many LCA queries on a static tree: binary lifting (O(log n) per query) or Euler tour + RMQ.
follow-up: Distance between two nodes = depth(p) + depth(q) − 2·depth(LCA).

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
