"""Interview coding problems (exercises/c*.py) and the re-solve schedule.

Each problem file starts with a docstring::

    title: Two Sum — indices of the pair with a target sum
    pattern: Arrays & hashing
    level: easy
    time: 15
    frequency: very high
    week: 1
    day: 2
    tags: hash-map
    complexity: O(n) time, O(n) space
    hint: What would you need to have seen already to finish at index i?
    hint: A dict from value to index answers that in O(1).
    follow-up: What if the array is sorted? (two pointers, O(1) space)

    Problem statement in Markdown ...

Repeated ``hint:`` / ``follow-up:`` lines become lists. Problems are scheduled two or three per week
alongside the main plan (``week`` / ``day``); the Coding interview page also lets you practise any of them,
timed, at any point.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .content import EX_DIR, SOL_DIR, TEST_DIR

PATTERNS = [
    "Arrays & hashing", "Two pointers", "Sliding window", "Stack", "Binary search", "Linked list", "Trees",
    "Trie", "Heap / priority queue", "Intervals", "Graphs", "Backtracking", "Dynamic programming", "Greedy",
    "Bit manipulation & math", "Matrix", "Design", "Randomised", "ML from scratch",
]
LEVELS = ["easy", "medium", "hard"]
FREQ = ["very high", "high", "medium"]

# Re-solve ladder (days until the next attempt) after a clean solve: tests pass, within the target time,
# no hints. A slow or hinted solve keeps the rung and comes back in 2 days; a failed attempt comes back tomorrow.
LADDER = [3, 10, 30, 90]


@dataclass
class Problem:
    key: str
    title: str
    pattern: str
    level: str
    minutes: int
    frequency: str
    week: int
    day: int
    tags: list[str]
    complexity: str
    hints: list[str]
    followups: list[str]
    spec: str
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def stub_path(self) -> Path:
        return EX_DIR / f"{self.key}.py"

    @property
    def solution_path(self) -> Path:
        return SOL_DIR / f"{self.key}.py"

    @property
    def test_path(self) -> Path:
        return TEST_DIR / f"test_{self.key}.py"

    @property
    def star(self) -> bool:
        return self.frequency == "very high"


def parse_problem(path: Path) -> Problem:
    src = path.read_text(encoding="utf-8")
    m = re.match(r'\s*"""(.*?)"""', src, re.S)
    doc = m.group(1).strip("\n") if m else ""
    head, _, spec = doc.partition("\n\n")
    meta: dict[str, str] = {}
    hints, follow = [], []
    for line in head.splitlines():
        if ":" not in line:
            continue
        k, v = (s.strip() for s in line.split(":", 1))
        if k == "hint":
            hints.append(v)
        elif k == "follow-up":
            follow.append(v)
        else:
            meta[k] = v
    known = {"title", "pattern", "level", "time", "frequency", "week", "day", "tags", "complexity"}
    return Problem(
        key=path.stem, title=meta.get("title", path.stem), pattern=meta.get("pattern", "Arrays & hashing"),
        level=meta.get("level", "medium"), minutes=int(meta.get("time", 25)), frequency=meta.get("frequency", "high"),
        week=int(meta.get("week", 0)), day=int(meta.get("day", 2)),
        tags=[t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
        complexity=meta.get("complexity", ""), hints=hints, followups=follow, spec=spec.strip(),
        extra={k: v for k, v in meta.items() if k not in known},
    )


@lru_cache(maxsize=1)
def problems() -> dict[str, Problem]:
    return {p.stem: parse_problem(p) for p in sorted(SOL_DIR.glob("c*.py"))}


def for_day(week: int, day: int) -> list[Problem]:
    return [p for p in problems().values() if p.week == week and p.day == day]


# ---- attempts & re-solve schedule ---------------------------------------------

def state(data: dict) -> dict:
    return data.setdefault("coding", {})


def record(data: dict, key: str, passed: bool, minutes: float, hints: int, today: dt.date | None = None) -> dict:
    """Log an attempt and move the problem on the re-solve ladder. Returns the updated entry."""
    today = today or dt.date.today()
    p = problems().get(key)
    target = p.minutes if p else 30
    e = state(data).setdefault(key, {"attempts": [], "rung": 0})
    e["attempts"].append({"date": today.isoformat(), "passed": passed, "minutes": round(minutes, 1), "hints": hints})
    if not passed:
        nxt = 1
    elif minutes <= target and hints == 0:
        nxt = LADDER[min(e["rung"], len(LADDER) - 1)]
        e["rung"] = min(e["rung"] + 1, len(LADDER))
    else:
        nxt = 2
    e["next"] = (today + dt.timedelta(days=nxt)).isoformat()
    return e


def solved(data: dict, key: str) -> bool:
    return any(a["passed"] for a in state(data).get(key, {}).get("attempts", []))


def best_minutes(data: dict, key: str) -> float | None:
    ts = [a["minutes"] for a in state(data).get(key, {}).get("attempts", []) if a["passed"]]
    return min(ts) if ts else None


def due(data: dict, today: dt.date | None = None) -> list[Problem]:
    """Problems whose re-solve date has arrived, most overdue first."""
    today = today or dt.date.today()
    out = []
    for key, e in state(data).items():
        if key in problems() and e.get("next") and dt.date.fromisoformat(e["next"]) <= today:
            out.append((e["next"], problems()[key]))
    return [p for _, p in sorted(out, key=lambda t: t[0])]


def mastered(data: dict, key: str) -> bool:
    """Solved cleanly at least twice in a row on the ladder."""
    return state(data).get(key, {}).get("rung", 0) >= 2
