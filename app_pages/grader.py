"""Answer grader: type your answer, get it graded against the reference like an interviewer would."""
import datetime as dt
import random

import streamlit as st

from mlprep import content, errorlog, grader, plan, progress, srs
from mlprep.ui import data, md

d = data()
today = dt.date.today()
cfg = d.setdefault("grader", {"backend": "offline", "model": "", "base_url": "http://localhost:8080/v1"})

st.title("Answer grader")
st.markdown(
    "Self-scoring is generous. Type (or paste a transcript of) your spoken answer and get it graded against the reference "
    "answer: what you covered, what you missed, what was wrong or vague, and the follow-up an interviewer would ask next. "
    "The grade can update the question's review schedule."
)

with st.expander("Grader backend", expanded=cfg["backend"] == "offline"):
    b = st.radio("Backend", ["offline", "local", "claude"], index=["offline", "local", "claude"].index(cfg["backend"]),
                 horizontal=True, format_func={"offline": "Offline key-concept check (no model)",
                                               "local": "Local model (llama.cpp / Ollama, OpenAI-compatible)",
                                               "claude": "Claude API"}.get)
    if b == "local":
        cfg["base_url"] = st.text_input("Base URL", cfg.get("base_url") or "http://localhost:8080/v1",
                                        help="llama.cpp: `llama-server -m model.gguf --port 8080`; Ollama: http://localhost:11434/v1")
        cfg["model"] = st.text_input("Model name", cfg.get("model", ""), help="e.g. qwen3.5:9b for Ollama; any string for llama-server")
    elif b == "claude":
        cfg["model"] = st.text_input("Model", cfg.get("model") or "claude-sonnet-5-5")
        st.caption("Reads the API key from the ANTHROPIC_API_KEY environment variable when the app starts; the key is "
                   "never written to disk by this app.")
    else:
        st.caption("Checks coverage of distinctive terms from the reference answer. It can't judge correctness — use a model "
                   "backend for real feedback.")
    if b != cfg["backend"]:
        cfg["backend"] = b
    progress.save(d)

start = progress.start_date(d)
pool = plan.unlocked_questions(start, today) + errorlog.as_questions(d)
mode = st.radio("Pick a question", ["Due for review", "Random ★ (unlocked)", "Choose a week"], horizontal=True)
if mode == "Choose a week":
    wk = st.selectbox("Week", list(content.weeks()), format_func=lambda k: f"W{k:02d} · {content.weeks()[k].title}")
    choices = content.weeks()[wk].questions
elif mode == "Due for review":
    choices = [q for q in pool if srs.is_due(d["cards"].get(q.qid), today)] or pool
else:
    choices = [q for q in pool if q.core] or pool
if not choices:
    st.info("No questions unlocked yet — choose a week instead.")
    st.stop()
if st.session_state.get("grader_mode") != mode or st.button("Another question", icon=":material/shuffle:"):
    st.session_state.grader_mode = mode
    st.session_state.grader_qid = random.choice(choices).qid
qidx = {q.qid: q for q in content.all_questions() + errorlog.as_questions(d)}
qid = st.session_state.get("grader_qid")
if qid not in qidx or (mode == "Choose a week" and qidx[qid] not in choices):
    qid = st.session_state.grader_qid = choices[0].qid
q = qidx[qid]

with st.container(border=True):
    st.caption(f"Week {q.week}" + (" · ★" if q.core else ""))
    st.markdown(f"### {q.text}")
ans = st.text_area("Your answer (aim for what you'd say in 60–120 seconds)", key=f"ans-{qid}", height=200)
if st.button("Grade my answer", type="primary", disabled=not ans.strip()):
    with st.spinner("Grading…"):
        try:
            res = grader.grade(q.text, q.answer, ans, cfg["backend"], cfg.get("model", ""), cfg.get("base_url", ""))
        except Exception as e:  # network/model errors shouldn't crash the page
            st.error(f"Grader error: {e}. Falling back to the offline check.")
            res = grader.offline_grade(q.text, q.answer, ans)
    st.session_state[f"res-{qid}"] = res
    d.setdefault("grades", []).append({"date": today.isoformat(), "qid": qid, "score": res["score"], "backend": res["backend"]})
    progress.save(d)

res = st.session_state.get(f"res-{qid}")
if res:
    st.metric("Score", f"{res['score']} / 3", help="0 wrong/blank · 1 partial · 2 solid · 3 strong with specifics")
    c1, c2 = st.columns(2)
    c1.markdown("**Covered**\n" + "\n".join(f"- {x}" for x in res["covered"]) if res["covered"] else "**Covered:** —")
    c2.markdown("**Missed**\n" + "\n".join(f"- {x}" for x in res["missed"]) if res["missed"] else "**Missed:** —")
    if res["errors"]:
        st.markdown("**Technically wrong**\n" + "\n".join(f"- {x}" for x in res["errors"]))
    if res["vague"]:
        st.markdown("**Vague**\n" + "\n".join(f"- {x}" for x in res["vague"]))
    if res.get("follow_up"):
        st.info(f"**Follow-up:** {res['follow_up']}")
    with st.expander("Reference answer"):
        md(q.answer)
    c1, c2 = st.columns(2)
    if c1.button("Apply grade to my review schedule", icon=":material/replay:"):
        d["cards"][qid] = srs.review(d["cards"].get(qid) or srs.new_card(), grader.RATING_FROM_SCORE[res["score"]], today)
        progress.save(d)
        st.success(f"Next review: {d['cards'][qid]['due']}")
    if res["score"] <= 1 and c2.button("Add to my error log", icon=":material/bug_report:"):
        errorlog.add(d, "Question I fumbled", q.text, q.answer, q.week, q.tags, in_review=False)
        progress.save(d)
        st.success("Added to the error log.")
