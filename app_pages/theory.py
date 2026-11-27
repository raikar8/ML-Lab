import datetime as dt

import streamlit as st

from mlprep import content, plan, progress
from mlprep.content import PHASES
from mlprep.ui import data, go_exercise, lab_link, md, question_expander

d = data()
weeks = content.weeks()
start = progress.start_date(d)
cur = plan.week_day_for(start, dt.date.today())

default_week = st.session_state.pop("theory_week", None) or (cur[0] if cur else 1)
default_day = st.session_state.pop("theory_day", None)

week = st.sidebar.selectbox("Week", list(weeks), index=list(weeks).index(default_week),
                            format_func=lambda k: f"W{k:02d} · {weeks[k].title}", key="theory_week_sel")
w = weeks[week]
st.caption(f"Phase {w.phase} · {PHASES[w.phase][0]} · Week {week} · starts {plan.date_for(start, week, 1):%d %b %Y}")
st.title(w.title)
if w.goal:
    st.markdown(f"**Goal.** {w.goal}")
if w.lab in plan.LABS:
    lab_link(w.lab, f"Lab for this week: {plan.LABS[w.lab][0]}")

options = [f"D{s.day} · {s.title}" for s in w.days] + ["Mini-project", "All questions"]
idx = 0
if default_day and default_day <= len(w.days):
    idx = default_day - 1
section = st.segmented_control("Section", options, default=options[idx], key=f"sec-{week}-{idx}",
                               label_visibility="collapsed") or options[idx]

if section == "Mini-project":
    st.header("Mini-project (day 6)")
    md(w.project or "_No project this week — the lab is the day 6 task._")
elif section == "All questions":
    st.header("All questions this week")
    st.caption("Answer out loud first, then open. ★ = high-yield (also used by the interview sprint).")
    for q in w.questions:
        question_expander(q)
else:
    sec = w.days[options.index(section)]
    st.header(f"Day {sec.day} · {sec.title}")
    md(sec.body)
    exs = content.exercises_for(week, sec.day)
    qs = [q for q in w.questions if q.day == sec.day]
    if exs:
        st.subheader("Code it")
        for ex in exs:
            c1, c2 = st.columns([0.75, 0.25], vertical_alignment="center")
            c1.markdown(f"**{ex.title}** · {ex.level} · ~{ex.minutes} min  \n:gray[`exercises/{ex.key}.py`]")
            if c2.button("Open exercise", key=f"ex-{ex.key}", width="stretch"):
                go_exercise(ex.key)
    if qs:
        st.subheader("Check yourself")
        st.caption("Say your answer out loud (aim for 60–90 seconds), then open to compare.")
        for q in qs:
            question_expander(q)
