import datetime as dt
import random

import streamlit as st

from mlprep import content, errorlog, plan, progress, srs
from mlprep.ui import data, md

d = data()
today = dt.date.today()
start = progress.start_date(d)
weeks = content.weeks()

st.title("Review (spaced repetition)")
st.markdown(
    "Every interview question joins your deck on the day it's taught. Say the answer **out loud**, reveal, then "
    "rate yourself honestly. *Again* brings it back tomorrow; *Good* and *Easy* push it further out. "
    "Ten minutes a day here is what makes six months of material stick."
)

with st.sidebar:
    scope = st.radio("Deck", ["Unlocked so far", "Everything (preview)"],
                     help="'Everything' lets you study ahead; cards keep their schedule either way.")
    wk_sel = st.multiselect("Only these weeks", list(weeks), format_func=lambda k: f"W{k:02d}")
    new_limit = st.slider("New cards per session", 0, 50, 15)
    core_only = st.toggle("★ high-yield only", False)

pool = plan.unlocked_questions(start, today) if scope.startswith("Unlocked") else content.all_questions()
pool = pool + errorlog.as_questions(d)  # your error-log cards are always in the deck
if wk_sel:
    pool = [q for q in pool if q.week in wk_sel]
if core_only:
    pool = [q for q in pool if q.core]

due = [q for q in pool if q.qid in d["cards"] and srs.is_due(d["cards"][q.qid], today)]
new = [q for q in pool if q.qid not in d["cards"]][:new_limit]

c1, c2, c3 = st.columns(3)
c1.metric("Due now", len(due))
c2.metric("New this session", len(new))
mast = [srs.mastery(d["cards"].get(q.qid)) for q in pool]
c3.metric("Solid or mastered", f"{sum(m in ('solid', 'mastered') for m in mast)}/{len(pool)}")

sig = (scope, tuple(wk_sel), new_limit, core_only, today.isoformat())
if st.session_state.get("rev_sig") != sig:
    queue = [q.qid for q in due]
    random.Random(today.toordinal()).shuffle(queue)
    st.session_state.rev_queue = queue + [q.qid for q in new]
    st.session_state.rev_sig = sig
    st.session_state.rev_show = False
queue = st.session_state.rev_queue

if not queue:
    st.success("Nothing due. Come back tomorrow, or widen the deck in the sidebar.")
    st.stop()

qidx = content.question_index() | {q.qid: q for q in errorlog.as_questions(d)}
q = qidx[queue[0]]
card = d["cards"].get(q.qid)
with st.container(border=True):
    st.caption(f"Week {q.week} · {weeks[q.week].title} · {srs.mastery(card)}"
               + (" · ★" if q.core else "") + f" · {len(queue)} left")
    st.markdown(f"### {q.text}")
    if not st.session_state.rev_show:
        if st.button("Show answer", type="primary"):
            st.session_state.rev_show = True
            st.rerun()
    else:
        md(q.answer)
        cols = st.columns(4)
        for r, name in srs.RATINGS.items():
            if cols[r].button(name, key=f"rate-{r}", width="stretch", type="primary" if r == 2 else "secondary"):
                d["cards"][q.qid] = srs.review(d["cards"].get(q.qid) or srs.new_card(), r, today)
                progress.save(d)
                queue.pop(0)
                if r == 0:
                    queue.append(q.qid)  # see it again this session
                st.session_state.rev_show = False
                st.rerun()
