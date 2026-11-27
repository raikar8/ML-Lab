"""Diagnostic quiz and the optional Week 0 (foundations) track."""
import datetime as dt

import pandas as pd
import streamlit as st

from mlprep import content, diagnostic, progress
from mlprep.ui import data, go_exercise, go_theory

d = data()
f0 = content.foundations().get("F0")

st.title("Diagnostic & Week 0")
st.markdown(
    "Twenty-four quick questions, four per area, on the background the 27-week plan assumes. Answer honestly, "
    "without looking anything up; it takes about 10 minutes. You'll get a recommendation per area: **skip**, "
    "**skim** or **study** the matching day of **Week 0**, the optional foundations week to do before week 1."
)

past = d.get("diagnostic", [])
if past:
    last = past[-1]
    st.caption(f"Last taken {last['date']}: {last['total']}/{len(diagnostic.QUESTIONS)} correct.")

with st.form("diag"):
    answers = {}
    for area in diagnostic.AREAS:
        st.subheader(area)
        for i, q in enumerate(diagnostic.QUESTIONS):
            if q.area != area:
                continue
            choice = st.radio(q.text, q.options, index=None, key=f"dq-{i}")
            answers[i] = q.options.index(choice) if choice is not None else None
    submitted = st.form_submit_button("Score my answers", type="primary")

if submitted:
    scores = diagnostic.score(answers)
    rec = diagnostic.recommend(scores)
    total = sum(c for c, _ in scores.values())
    d.setdefault("diagnostic", []).append({
        "date": dt.date.today().isoformat(), "total": total,
        "scores": {a: list(s) for a, s in scores.items()}, "study_days": rec["study_days"]})
    progress.save(d)
    st.session_state["diag_result"] = (answers, scores, rec)

if "diag_result" in st.session_state:
    answers, scores, rec = st.session_state["diag_result"]
    st.header(f"Result: {sum(c for c, _ in scores.values())} / {len(diagnostic.QUESTIONS)}")
    icon = {"skip": "✅ skip", "skim": "🟡 skim", "study": "🔴 study"}
    st.dataframe(pd.DataFrame([{"Area": a, "Score": f"{scores[a][0]}/{scores[a][1]}", "Recommendation": icon[p["level"]],
                                "Where": p["what"]} for a, p in rec["areas"].items()]),
                 hide_index=True, width="stretch")
    if rec["skip_week0"]:
        st.success("You can skip Week 0 and start at week 1. " + rec["coding"])
    else:
        days = ", ".join(f"D{x}" for x in rec["study_days"])
        skim = (" and skim " + ", ".join(f"D{x}" for x in rec["skim_days"])) if rec["skim_days"] else ""
        st.warning(f"Do Week 0 {days}{skim} before week 1 (about 2 hours per day, including the exercise). "
                   + rec["coding"])
    with st.expander("Review the questions you missed"):
        for i, q in enumerate(diagnostic.QUESTIONS):
            if answers.get(i) != q.answer:
                given = q.options[answers[i]] if answers.get(i) is not None else "—"
                st.markdown(f"**{q.text}**  \nYour answer: {given} · Correct: **{q.options[q.answer]}**  \n:gray[{q.why}]")

st.divider()
st.header("Week 0 — Foundations")
if f0 is None:
    st.info("Week 0 content not found.")
    st.stop()
st.markdown(f0.goal)
exs = {e.day: e for e in content.exercises_for("F0")}
for sec in f0.days:
    c1, c2, c3 = st.columns([0.6, 0.2, 0.2], vertical_alignment="center")
    c1.markdown(f"**D{sec.day} · {sec.title}**" + (f"  \n:gray[Exercise: {exs[sec.day].title}]" if sec.day in exs else ""))
    if c2.button("Read", key=f"f0-read-{sec.day}", width="stretch"):
        go_theory("F0", sec.day)
    if sec.day in exs and c3.button("Code it", key=f"f0-ex-{sec.day}", width="stretch"):
        go_exercise(exs[sec.day].key)
st.caption("Week 0's questions can join your review deck: turn on 'Include Week 0 & electives' on the Review page.")
