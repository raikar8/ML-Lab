"""A small SM-2 style spaced-repetition scheduler for the question bank.

A card is introduced on the day its week/day is scheduled. After each review you rate it:
0 = Again (forgot), 1 = Hard, 2 = Good, 3 = Easy. The interval grows with the ease factor.
"""
from __future__ import annotations

import datetime as dt

RATINGS = {0: "Again", 1: "Hard", 2: "Good", 3: "Easy"}


def new_card() -> dict:
    return {"ease": 2.5, "interval": 0, "reps": 0, "due": None, "lapses": 0, "history": []}


def review(card: dict, rating: int, today: dt.date) -> dict:
    """Return the updated card (mutates and returns it)."""
    ease, interval, reps = card["ease"], card["interval"], card["reps"]
    if rating == 0:
        reps, interval = 0, 1
        card["lapses"] += 1
        ease = max(1.3, ease - 0.2)
    else:
        if reps == 0:
            interval = 1 if rating == 1 else (2 if rating == 2 else 4)
        elif reps == 1:
            interval = 3 if rating == 1 else (6 if rating == 2 else 8)
        else:
            mult = {1: 1.2, 2: ease, 3: ease * 1.3}[rating]
            interval = max(interval + 1, round(interval * mult))
        reps += 1
        ease = max(1.3, ease + {1: -0.15, 2: 0.0, 3: 0.15}[rating])
    card.update(ease=round(ease, 3), interval=interval, reps=reps,
                due=(today + dt.timedelta(days=interval)).isoformat())
    card["history"] = (card.get("history", []) + [[today.isoformat(), rating]])[-20:]
    return card


def is_due(card: dict | None, today: dt.date) -> bool:
    if card is None or card.get("due") is None:
        return True
    return dt.date.fromisoformat(card["due"]) <= today


def mastery(card: dict | None) -> str:
    if not card or card["reps"] == 0:
        return "new"
    if card["interval"] >= 21:
        return "mastered"
    if card["interval"] >= 6:
        return "solid"
    return "learning"
