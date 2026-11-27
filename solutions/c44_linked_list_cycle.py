"""
title: Linked-list cycle — detect it, find where it starts, and find the middle
pattern: Linked list
level: medium
time: 25
frequency: very high
week: 22
day: 4
tags: linked-list, fast-slow-pointers, floyd
complexity: O(n) time, O(1) space
hint: Floyd: slow moves 1, fast moves 2; they meet inside the cycle iff there is one.
hint: After meeting, restart one pointer at head; moving both one step at a time, they meet at the cycle start.
follow-up: Prove the restart trick (distance to start ≡ distance from meeting point to start, mod cycle length).
follow-up: Find the duplicate number in an array of n+1 values in [1, n] in O(1) space — it's the same algorithm.

`ListNode`, `from_list` are given (and `make_cycle(head, pos)` that links the tail to node `pos`).

* `has_cycle(head) -> bool`;
* `cycle_start(head)` — the node where the cycle begins, or `None`;
* `middle(head)` — the middle node of an acyclic list (the second middle for even lengths), `None` if empty;
* `find_duplicate(nums)` — `nums` has `n + 1` values in `[1, n]` and exactly one value repeated (possibly many
  times); return it in O(1) extra space without modifying `nums`.

All with O(1) extra space (no `set` of visited nodes).
"""
from __future__ import annotations


# given
class ListNode:
    def __init__(self, val=0, next: ListNode | None = None):
        self.val, self.next = val, next


# given
def from_list(vals):
    head = None
    for v in reversed(vals):
        head = ListNode(v, head)
    return head


# given
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
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            p = head
            while p is not slow:
                p, slow = p.next, slow.next
            return p
    return None


def has_cycle(head) -> bool:
    return cycle_start(head) is not None


def middle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow


def find_duplicate(nums: list[int]) -> int:
    slow = fast = nums[0]
    while True:
        slow, fast = nums[slow], nums[nums[fast]]
        if slow == fast:
            break
    p = nums[0]
    while p != slow:
        p, slow = nums[p], nums[slow]
    return p
