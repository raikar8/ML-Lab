import datetime as dt
import random
import time

import pandas as pd
import streamlit as st

from mlprep import content, plan, progress
from mlprep.content import PHASES
from mlprep.ui import data, md

d = data()
today = dt.date.today()
start = progress.start_date(d)
weeks = content.weeks()

FOLLOW_UPS = [
    "How would this change on a microcontroller with 256 KB of RAM?",
    "What's the computational and memory cost, in big-O and in real numbers?",
    "What's the most common bug or failure mode here, and how would you detect it?",
    "Give a concrete example from a project you've done.",
    "What would you try first if this didn't work, and why?",
    "How would you explain this to a product manager in two sentences?",
    "What's the trade-off you're making, and when would you choose the other side?",
    "How would you test or validate this before shipping?",
    "What changed about this between 2017-era and 2026-era models?",
    "How does this interact with quantization?",
]
SCORES = {0: "0 · blanked", 1: "1 · partial", 2: "2 · solid", 3: "3 · strong + example"}

st.title("Mock interview")

if "mock" not in st.session_state:
    pre = st.session_state.pop("mock_week", None)
    st.markdown(
        "A timed question round. For each question: **talk for 1–3 minutes as if to an interviewer** (record "
        "yourself if you can), answer the follow-up, then reveal the model answer and score yourself. "
        "Scores are kept so you can see weak areas."
    )
    scopes = ["One week", "One phase", "Everything unlocked so far", "Everything"]
    scope = st.radio("Scope", scopes, index=0 if pre else 2, horizontal=True)
    if scope == "One week":
        wk = st.selectbox("Week", list(weeks), index=(pre - 1) if pre else 0,
                          format_func=lambda k: f"W{k:02d} · {weeks[k].title}")
        pool = weeks[wk].questions
        label = f"Week {wk}"
    elif scope == "One phase":
        ph = st.selectbox("Phase", list(PHASES), format_func=lambda p: f"{p} · {PHASES[p][0]}")
        pool = [q for w in weeks.values() if w.phase == ph for q in w.questions]
        label = f"Phase {ph}"
    elif scope.startswith("Everything unlocked"):
        pool = plan.unlocked_questions(start, today)
        label = "Unlocked"
    else:
        pool = content.all_questions()
        label = "All"
    n = st.slider("Questions", 3, 20, 8)
    secs = st.slider("Seconds per question (soft limit)", 60, 300, 150, step=30)
    prefer_core = st.toggle("Favour ★ high-yield questions", True)
    if not pool:
        st.warning("No questions in this scope yet.")
        st.stop()
    if st.button("Start mock", type="primary", icon=":material/play_arrow:"):
        rng = random.Random()
        core = [q for q in pool if q.core]
        rest = [q for q in pool if not q.core]
        rng.shuffle(core)
        rng.shuffle(rest)
        picked = (core + rest)[:n] if prefer_core else rng.sample(pool, min(n, len(pool)))
        rng.shuffle(picked)
        st.session_state.mock = {"label": label, "qids": [q.qid for q in picked], "i": 0, "scores": [],
                                 "secs": secs, "t0": time.time(), "show": False,
                                 "fu": [rng.choice(FOLLOW_UPS) for _ in picked]}
        st.rerun()
    if d["mocks"]:
        st.subheader("History")
        hist = pd.DataFrame([{"Date": m["date"], "Scope": m["label"], "Questions": len(m["scores"]),
                              "Score": m["pct"]} for m in d["mocks"]][::-1])
        st.dataframe(hist, hide_index=True, width="stretch",
                     column_config={"Score": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                            max_value=1)})
    st.stop()

m = st.session_state.mock
qidx = content.question_index()
if m["i"] >= len(m["qids"]):
    pct = sum(s for _, s in m["scores"]) / (3 * max(1, len(m["scores"])))
    st.success(f"Done — {pct:.0%} ({m['label']}).")
    weak = [qidx[q] for q, s in m["scores"] if s <= 1]
    if weak:
        st.markdown("**Revisit these** (they've been reset to *due today* on the Review page):")
        for q in weak:
            st.markdown(f"- W{q.week:02d}: {q.text}")
            if q.qid in d["cards"]:
                d["cards"][q.qid]["due"] = today.isoformat()
    d["mocks"].append({"date": today.isoformat(), "label": m["label"], "scores": m["scores"], "pct": round(pct, 3)})
    progress.save(d)
    del st.session_state.mock
    st.button("New mock")
    st.stop()

q = qidx[m["qids"][m["i"]]]
st.caption(f"Question {m['i'] + 1} of {len(m['qids'])} · {m['label']} · week {q.week}: {weeks[q.week].title}")


@st.fragment(run_every=1)
def timer():
    el = int(time.time() - m["t0"])
    over = el > m["secs"]
    st.markdown(f"⏱ **{el // 60}:{el % 60:02d}** / {m['secs'] // 60}:{m['secs'] % 60:02d}"
                + ("  :red[— wrap up]" if over else ""))


timer()
with st.container(border=True):
    st.markdown(f"### {q.text}")
    st.markdown(f"**Follow-up:** _{m['fu'][m['i']]}_")
if not m["show"]:
    if st.button("Reveal model answer", type="primary"):
        m["show"] = True
        st.rerun()
else:
    md(q.answer)
    sc = st.radio("Score yourself", list(SCORES), format_func=SCORES.get, horizontal=True, index=None)
    if st.button("Next", type="primary", disabled=sc is None):
        m["scores"].append([q.qid, sc])
        m["i"] += 1
        m["show"] = False
        m["t0"] = time.time()
        st.rerun()
