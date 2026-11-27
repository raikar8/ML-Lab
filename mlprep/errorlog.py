"""Personal error log: bugs you hit and questions you fumbled, turned into review cards."""
from __future__ import annotations

import datetime as dt

from .content import Question

KINDS = ["Question I fumbled", "Bug I made", "Concept gap", "Interview question I was asked"]


def add(data: dict, kind: str, prompt: str, lesson: str, week: int | None, tags: list[str], in_review: bool) -> dict:
    log = data.setdefault("errors", [])
    eid = max((e["id"] for e in log), default=0) + 1
    entry = {"id": eid, "date": dt.date.today().isoformat(), "kind": kind, "prompt": prompt.strip(),
             "lesson": lesson.strip(), "week": week, "tags": tags, "in_review": in_review}
    log.append(entry)
    return entry


def as_questions(data: dict) -> list[Question]:
    out = []
    for e in data.get("errors", []):
        if e.get("in_review", True) and e["prompt"] and e["lesson"]:
            out.append(Question(f"err-{e['id']}", e.get("week") or 0, 0, f"[{e['kind']}] {e['prompt']}", e["lesson"],
                                list(e.get("tags", [])) + ["error-log"], core=True))
    return out
