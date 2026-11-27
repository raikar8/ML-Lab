"""
title: Meeting rooms — minimum rooms, and the busiest moment
pattern: Intervals
level: medium
time: 25
frequency: very high
week: 14
day: 4
tags: intervals, heap, sweep-line
complexity: O(n log n) time, O(n) space
hint: Sweep line: +1 at each start, −1 at each end; the running maximum is the answer. Process ends before starts at equal times.
hint: Or: sort by start and keep a min-heap of end times of rooms in use; reuse the room that frees up first.
follow-up: Assign an actual room number to each meeting (heap of (end, room)).
follow-up: Same idea for the peak number of concurrent inference requests in a log → capacity planning.

Meetings are half-open intervals `[start, end)` (a meeting ending at 10 frees the room for one starting at 10).

* `min_rooms(meetings)` — minimum number of rooms so no two overlapping meetings share a room;
* `can_attend_all(meetings) -> bool` — True if no two meetings overlap;
* `events(meetings)` — the sorted sweep-line events `(time, +1|−1)` with ends before starts at equal times;
* `busiest(meetings)` — `(time, count)`: the **earliest** time at which the number of simultaneous meetings
  reaches its maximum (for an empty list return `(None, 0)`).

```
min_rooms([(0, 30), (5, 10), (15, 20)]) -> 2
busiest([(1, 4), (2, 5), (4, 6)])       -> (2, 2)
```
"""


def events(meetings) -> list[tuple[int, int]]:
    """Sorted sweep events (time, +1 for a start / −1 for an end); ends come first at equal times."""
    ev = []
    for s, e in meetings:
        ev.append((s, 1))
        ev.append((e, -1))
    ev.sort()  # at equal times -1 (end) sorts before +1 (start)
    return ev


def busiest(meetings):
    best_t, best, cur = None, 0, 0
    for t, d in events(meetings):
        cur += d
        if cur > best:
            best, best_t = cur, t
    return best_t, best


def min_rooms(meetings) -> int:
    return busiest(meetings)[1]


def can_attend_all(meetings) -> bool:
    return min_rooms(meetings) <= 1
