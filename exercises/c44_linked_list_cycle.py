"""
Linked-list cycle — detect it, find where it starts, and find the middle

`ListNode`, `from_list` are given (and `make_cycle(head, pos)` that links the tail to node `pos`).

* `has_cycle(head) -> bool`;
* `cycle_start(head)` — the node where the cycle begins, or `None`;
* `middle(head)` — the middle node of an acyclic list (the second middle for even lengths), `None` if empty;
* `find_duplicate(nums)` — `nums` has `n + 1` values in `[1, n]` and exactly one value repeated (possibly many
  times); return it in O(1) extra space without modifying `nums`.

All with O(1) extra space (no `set` of visited nodes).
"""

from __future__ import annotations

class ListNode:
    def __init__(self, val=0, next: ListNode | None = None):
        self.val, self.next = val, next

def from_list(vals):
    head = None
    for v in reversed(vals):
        head = ListNode(v, head)
    return head

def make_cycle(head, pos):
    """Link the tail to the node at index pos (pos = -1: no cycle). Returns head."""
    if head is None or pos < 0:
        return head
    nodes = []
    cur = head
    while cur:
        nodes.append(cur)
        cur = cur.next
    nodes[-1].next = nodes[pos]
    return head

def cycle_start(head):
    raise NotImplementedError  # TODO: implement


def has_cycle(head) -> bool:
    raise NotImplementedError  # TODO: implement


def middle(head):
    raise NotImplementedError  # TODO: implement


def find_duplicate(nums: list[int]) -> int:
    raise NotImplementedError  # TODO: implement
