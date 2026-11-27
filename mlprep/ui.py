"""Shared Streamlit helpers."""
from __future__ import annotations

import re

import streamlit as st

from . import content, papers, progress
from .plan import LABS, Task

PAGES = {
    "today": "app_pages/today.py",
    "roadmap": "app_pages/roadmap.py",
    "theory": "app_pages/theory.py",
    "exercises": "app_pages/exercises.py",
    "review": "app_pages/review.py",
    "mock": "app_pages/mock.py",
    "sprint": "app_pages/sprint.py",
    "atlas": "app_pages/atlas.py",
    "transformers": "app_pages/transformers.py",
    "frontier": "app_pages/frontier.py",
    "settings": "app_pages/settings.py",
    "cpp": "app_pages/cpp.py",
    "grader": "app_pages/grader.py",
    "errorlog": "app_pages/errorlog.py",
    "designs": "app_pages/designs.py",
    "capstone": "app_pages/capstone.py",
    "notebooks": "app_pages/notebooks.py",
    "papers": "app_pages/papers.py",
    "coding": "app_pages/coding.py",
    "diagnostic": "app_pages/diagnostic.py",
}

KIND_ICON = {"theory": ":material/menu_book:", "exercise": ":material/code:", "coding": ":material/terminal:", "quiz": ":material/quiz:",
             "lab": ":material/science:", "project": ":material/build:", "review": ":material/replay:",
             "mock": ":material/timer:", "note": ":material/edit_note:"}

_MATH = re.compile(r"```math\n(.*?)```", re.S)


def md(text: str) -> None:
    """Render Markdown; ```math fenced blocks become st.latex (one equation per block)."""
    pos = 0
    for m in _MATH.finditer(text):
        chunk = text[pos:m.start()].strip()
        if chunk:
            st.markdown(chunk)
        st.latex(m.group(1).strip())
        pos = m.end()
    rest = text[pos:].strip()
    if rest:
        st.markdown(rest)


def data() -> dict:
    if "progress" not in st.session_state:
        st.session_state.progress = progress.load()
    return st.session_state.progress


def go_theory(week: int, day: int | None = None) -> None:
    st.session_state.theory_week = week
    st.session_state.theory_day = day
    st.switch_page(PAGES["theory"])


def go_exercise(key: str) -> None:
    st.session_state.exercise_key = key
    st.switch_page(PAGES["exercises"])


def go_papers(section: str | None) -> None:
    """Open the Papers page filtered to one week ('W12') or elective ('E1')."""
    st.session_state.papers_section = section
    st.switch_page(PAGES["papers"])


def go_coding(key: str | None) -> None:
    """Open the Coding interview page on one problem."""
    st.session_state.coding_key = key
    st.switch_page(PAGES["coding"])


def go_mock(week: int | None) -> None:
    st.session_state.mock_week = week
    st.switch_page(PAGES["mock"])


def task_row(t: Task, key_prefix: str = "") -> None:
    d = data()
    c1, c2, c3 = st.columns([0.07, 0.68, 0.25], vertical_alignment="center")
    done = c1.checkbox("done", value=progress.is_done(d, t.tid), key=f"{key_prefix}{t.tid}",
                       label_visibility="collapsed")
    if done != progress.is_done(d, t.tid):
        progress.set_done(d, t.tid, done)
    label = f"~~{t.title}~~" if done else t.title
    c2.markdown(f"{KIND_ICON.get(t.kind, '')} {label}  \n:gray[{t.minutes} min · week {t.week}, day {t.day}]")
    k = f"{key_prefix}go-{t.tid}"
    if t.kind in ("theory", "quiz", "project", "note"):
        if c3.button("Open", key=k, width="stretch"):
            go_theory(t.week, t.day if t.kind in ("theory", "quiz") else None)
    elif t.kind == "exercise":
        if c3.button("Open", key=k, width="stretch"):
            go_exercise(t.ref)
    elif t.kind == "coding":
        if c3.button("Solve", key=k, width="stretch"):
            go_coding(t.ref)
    elif t.kind == "lab":
        lab_link(t.ref, "Open lab", c3)
    elif t.kind == "review":
        c3.page_link(PAGES["review"], label="Review", icon=":material/replay:")
    elif t.kind == "mock":
        if c3.button("Start", key=k, width="stretch"):
            go_mock(t.week)


def question_expander(q: content.Question, prefix: str = "") -> None:
    star = " ★" if q.core else ""
    with st.expander(f"{q.text}{star}"):
        md(q.answer)
        if q.tags:
            st.caption("Tags: " + ", ".join(q.tags))


def lab_link(key: str, label: str | None = None, where=None) -> None:
    """Page link to a lab, if that lab page exists."""
    title, path, _ = LABS[key]
    if (content.ROOT / path).exists():
        (where or st).page_link(path, label=label or f"Lab: {title}", icon=":material/science:")
    else:
        (where or st).caption(f"Lab '{title}' coming soon")


def sources(titles: list[str], label: str = "Primary sources", where=None) -> None:
    """Expander listing reading-list papers by exact title (links come from content/reading.md)."""
    with (where or st).expander(f"{label} ({len(titles)})", icon=":material/article:"):
        st.markdown(papers.refs_markdown(titles))
        st.caption("Tick them off and write notes on the Papers page.")
