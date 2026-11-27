"""
Meeting rooms — minimum rooms, and the busiest moment

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
    raise NotImplementedError  # TODO: implement


def busiest(meetings):
    raise NotImplementedError  # TODO: implement


def min_rooms(meetings) -> int:
    raise NotImplementedError  # TODO: implement


def can_attend_all(meetings) -> bool:
    raise NotImplementedError  # TODO: implement
