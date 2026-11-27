import datetime as dt

import pandas as pd
import streamlit as st

from mlprep import content, plan, progress, srs
from mlprep.content import PHASES
from mlprep.ui import data, go_theory

d = data()
start = progress.start_date(d)
today = dt.date.today()
cur = plan.week_day_for(start, today)

st.title("6-month roadmap")
st.markdown(
    "27 weeks in six phases. The order is deliberate: fundamentals that every loop tests come first, then the "
    "deep-learning and transformer core, then the modern LLM techniques (where the Chinese labs led), and the "
    "embedded / optimisation block sits late so it's freshest for embedded-AI interviews. The last three weeks "
    "are system design, timed coding and full mocks."
)

rows = []
for wk, w in content.weeks().items():
    tasks = [t for dd in range(1, 8) for t in plan.tasks_for(wk, dd)]
    done = sum(progress.is_done(d, t.tid) for t in tasks)
    cards = [d["cards"].get(q.qid) for q in w.questions]
    solid = sum(srs.mastery(c) in ("solid", "mastered") for c in cards)
    rows.append({"Week": wk, "Phase": PHASES[w.phase][0], "Starts": plan.date_for(start, wk, 1),
                 "Topic": w.title, "Lab": plan.LABS[w.lab][0] if w.lab in plan.LABS else "project only",
                 "Exercises": len(content.exercises_for(wk)), "Tasks done": done / max(1, len(tasks)),
                 "Cards solid": solid / max(1, len(cards))})
df = pd.DataFrame(rows)

for ph, (name, blurb) in PHASES.items():
    sub = df[df.Phase == name]
    if sub.empty:
        continue
    st.subheader(f"Phase {ph} · {name}")
    st.caption(f"Weeks {sub.Week.min()}–{sub.Week.max()} · {blurb}")
    st.dataframe(
        sub.drop(columns=["Phase"]), hide_index=True, width="stretch",
        column_config={
            "Starts": st.column_config.DateColumn(format="ddd D MMM YYYY"),
            "Tasks done": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Cards solid": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                                           help="Share of the week's questions you've reviewed "
                                                                "to an interval of 6+ days."),
        })

st.divider()
if cur:
    st.markdown(f"You are in **week {cur[0]}**, day {cur[1]}.")
pick = st.selectbox("Open a week", list(content.weeks()), format_func=lambda k: f"Week {k}: "
                    f"{content.weeks()[k].title}", index=(cur[0] - 1) if cur else 0)
if st.button("Open theory for this week", icon=":material/menu_book:"):
    go_theory(pick)
