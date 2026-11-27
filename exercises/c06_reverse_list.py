"""
title: Reverse a linked list — whole list, and k nodes at a time
pattern: Linked list
level: easy
time: 25
frequency: very high
week: 3
day: 4
tags: linked-list, pointers
complexity: O(n) time, O(1) extra space (iterative)
hint: Walk the list with prev / cur; at each step save cur.next, point cur.next at prev, then advance both.
hint: For k-groups: check that k nodes remain, reverse exactly those, and connect the old group head to the next result.
follow-up: Do the whole-list reversal recursively — what is the space cost?
follow-up: Reverse only positions left..right (1-indexed) in one pass.

`ListNode`, `from_list` and `to_list` are given. Implement

* `reverse(head)` — reverse the list in place and return the new head;
* `reverse_k_group(head, k)` — reverse every consecutive block of `k` nodes; a final block shorter than `k`
  stays as it is. Re-link the existing nodes (don't create new ones or swap values).

```
to_list(reverse(from_list([1, 2, 3])))              -> [3, 2, 1]
to_list(reverse_k_group(from_list([1,2,3,4,5]), 2)) -> [2, 1, 4, 3, 5]
```
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

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def reverse(head: ListNode | None) -> ListNode | None:
    raise NotImplementedError  # TODO: implement


def reverse_k_group(head: ListNode | None, k: int) -> ListNode | None:
    raise NotImplementedError  # TODO: implement
