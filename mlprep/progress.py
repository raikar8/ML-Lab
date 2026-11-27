"""Progress is a small JSON file next to the app (git-ignored), so it survives restarts.

Set MLPREP_PROGRESS to keep it somewhere else (e.g. a synced folder).
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from .content import ROOT

PATH = Path(os.environ.get("MLPREP_PROGRESS", ROOT / "progress.json"))


def _default() -> dict:
    return {"start_date": dt.date.today().isoformat(), "done": {}, "cards": {}, "mocks": [], "notes": {},
            "exercise_status": {}, "sprint": {}, "hours_per_day": 2.0}


def load() -> dict:
    if PATH.exists():
        try:
            data = json.loads(PATH.read_text(encoding="utf-8"))
            base = _default()
            base.update(data)
            return base
        except (json.JSONDecodeError, OSError):
            pass
    data = _default()
    save(data)
    return data


def save(data: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(PATH)


def start_date(data: dict) -> dt.date:
    return dt.date.fromisoformat(data["start_date"])


def set_done(data: dict, task_id: str, done: bool) -> None:
    if done:
        data["done"][task_id] = dt.date.today().isoformat()
    else:
        data["done"].pop(task_id, None)
    save(data)


def is_done(data: dict, task_id: str) -> bool:
    return task_id in data["done"]
