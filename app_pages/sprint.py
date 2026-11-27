"""Interview in a week: a 7-day crash plan built from the high-yield (★) material for the chosen role."""
import datetime as dt

import streamlit as st

from mlprep import content, progress, srs
from mlprep.ui import data, go_exercise, go_theory, question_expander

d = data()
today = dt.date.today()
weeks = content.weeks()

TRACKS = {
    "ML Engineer (general)": {
        "weeks": [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 17, 20, 26, 27],
        "blurb": "Classical ML + deep learning fundamentals, PyTorch fluency, transformers, ML system design.",
    },
    "Embedded / Edge AI engineer": {
        "weeks": [5, 6, 7, 9, 10, 11, 12, 14, 21, 22, 23, 24, 25, 26],
        "blurb": "Model optimisation (quantization, pruning, distillation), hardware/roofline, deployment runtimes, "
                 "real-time pipelines.",
    },
    "LLM / GenAI engineer": {
        "weeks": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 27],
        "blurb": "Transformer internals, MoE, efficient attention, inference & serving, post-training (RLHF/DPO/GRPO), "
                 "frontier models.",
    },
    "Audio / Speech ML engineer": {
        "weeks": [2, 6, 7, 8, 9, 10, 16, 20, 21, 22, 24, 25, 26],
        "blurb": "Speech front-ends, sequence models & CTC, enhancement, streaming/real-time deployment, "
                 "quantization for DSPs.",
    },
}

# day template: (title, which slice of the track's weeks, activities)
TEMPLATE = [
    ("Fundamentals blitz", slice(0, 4), "Re-derive the core maths out loud; 2 must-code drills; 15 ★ questions."),
    ("Deep learning & PyTorch core", slice(4, 7), "Timed: MHA, norms, training loop from a blank file; 15 ★ questions."),
    ("Track specialty I", slice(7, 10), "Read the summaries of these weeks; 2 drills; 15 ★ questions."),
    ("Track specialty II + system design", slice(10, 14), "One 45-min system design out loud (week 26); 15 ★ questions."),
    ("Full mock loop", slice(0, 14), "Coding 45′ · fundamentals 30′ · depth 30′ · design 45′ · behavioural 30′. "
                                     "Score each round."),
    ("Weak spots + stories", slice(0, 14), "Redo the lowest-scored mock topics; rehearse 3 project stories and 4 STAR "
                                           "stories (2-minute versions)."),
    ("Light review & rest", slice(0, 14), "30–60 min of ★ cards only, logistics, sleep. No new material."),
]

st.title("Interview in a week")
st.markdown(
    "Got an interview coming up? Enter the date and the kind of role. You get a day-by-day crash plan drawn from the "
    "high-yield (★) questions and must-know exercises of the weeks that matter most for that role — weakest material "
    "first, based on your review history. It runs alongside (or instead of) the 6-month plan."
)

sp = d.setdefault("sprint", {})
c1, c2, c3 = st.columns([1, 1.4, 1])
date = c1.date_input("Interview date", dt.date.fromisoformat(sp["date"]) if sp.get("date") else today + dt.timedelta(days=7),
                     min_value=today)
track = c2.selectbox("Role focus", list(TRACKS), index=list(TRACKS).index(sp.get("track", list(TRACKS)[0])))
company = c3.text_input("Company (optional)", sp.get("company", ""))
st.caption(TRACKS[track]["blurb"])
if (date.isoformat(), track, company) != (sp.get("date"), sp.get("track"), sp.get("company")):
    sp.update(date=date.isoformat(), track=track, company=company)
    progress.save(d)

days_left = (date - today).days
n_days = max(1, min(7, days_left))
plan_days = TEMPLATE[-n_days:] if n_days < 7 else TEMPLATE
if days_left == 0:
    st.warning("The interview is today — see the day-of checklist at the bottom.")
else:
    st.info(f"**{days_left} day{'s' if days_left != 1 else ''} to go.** "
            + ("Using the full 7-day plan." if days_left >= 7 else
               f"Compressed to the last {n_days} day(s) of the plan."))

tw = TRACKS[track]["weeks"]


def weakness(q):
    c = d["cards"].get(q.qid)
    if not c or c["reps"] == 0:
        return (1, 0)
    return (0 if c["interval"] < 6 else 2, c["interval"])


def picks(wks, n_q=15, n_ex=2):
    qs = [q for w in wks for q in weeks[w].questions if q.core]
    qs.sort(key=weakness)
    exs = [e for w in wks for e in content.exercises_for(w) if e.core]
    exs.sort(key=lambda e: d["exercise_status"].get(e.key, {}).get("ok", False))
    return qs[:n_q], exs[:n_ex]


done = sp.setdefault("done", {})
for i, (title, sl, how) in enumerate(plan_days):
    day_date = date - dt.timedelta(days=len(plan_days) - i)
    wks = tw[sl] or tw  # the last three days (mock, stories, rest) use every week of the track
    key = f"{date.isoformat()}-{i}"
    label = f"{'✅' if done.get(key) else '⬜'} {day_date:%a %d %b} — {title}"
    with st.expander(label, expanded=(day_date == today)):
        st.markdown(f"**How:** {how}")
        st.caption("Weeks in focus: " + ", ".join(f"W{w} {weeks[w].title.split('—')[0].strip()}" for w in wks[:6])
                   + (" …" if len(wks) > 6 else ""))
        if title.startswith("Light"):
            qs, exs = picks(wks, n_q=10, n_ex=0)
        elif title.startswith("Full mock"):
            qs, exs = [], picks(wks, n_q=0, n_ex=2)[1]
        else:
            qs, exs = picks(wks)
        if exs:
            st.markdown("**Must-code drills** (blank file, timer on):")
            for ex in exs:
                a, b = st.columns([0.75, 0.25], vertical_alignment="center")
                ok = d["exercise_status"].get(ex.key, {}).get("ok")
                a.markdown(f"{'✅' if ok else '⬜'} {ex.title} · ~{ex.minutes} min · W{ex.week}")
                if b.button("Open", key=f"sp-ex-{key}-{ex.key}", width="stretch"):
                    go_exercise(ex.key)
        if title.startswith("Full mock"):
            st.page_link("app_pages/mock.py", label="Start a mock (scope: everything, ★ favoured, 10 questions)",
                         icon=":material/timer:")
        if title.startswith("Track specialty II"):
            if st.button("Open week 26 — system design", key=f"sp-sd-{key}"):
                go_theory(26)
        if title.startswith("Weak"):
            if st.button("Open week 27 — stories & behavioural", key=f"sp-beh-{key}"):
                go_theory(27, 4)
        if qs:
            st.markdown(f"**{len(qs)} ★ questions** — answer out loud first (weakest first):")
            for q in qs:
                question_expander(q, prefix=key)
        if st.checkbox("Done", value=bool(done.get(key)), key=f"sp-done-{key}"):
            if not done.get(key):
                done[key] = today.isoformat()
                progress.save(d)
        elif done.get(key):
            done.pop(key)
            progress.save(d)

st.divider()
st.subheader("Company & role research (do it on day 1)")
st.markdown(f"""
* What does **{company or 'the company'}** ship that uses ML? Which devices/platforms (MCU, DSP, phone NPU, car SoC, cloud)?
* Read the job description line by line; map each requirement to a story or a week in this app.
* Recent talks, papers, blog posts or open-source repos from the team — prepare one question about one of them.
* Ask the recruiter: number and type of rounds, coding language/environment, whether there's a system-design or
  take-home round, and who the interviewers are.
""")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Day before")
    for item in ["Test camera, mic, internet and IDE / shared-editor link", "Re-read the job description and your CV",
                 "Rehearse 3 project stories (2-minute versions) once, out loud",
                 "Prepare 3 questions for each interviewer", "Glass of water, notebook, pen ready",
                 "Stop studying by the evening; sleep 7–8 hours"]:
        st.checkbox(item, key=f"before-{item}")
with c2:
    st.subheader("Day of")
    for item in ["Light 20-minute warm-up: one easy exercise + 5 ★ cards", "Join 5 minutes early",
                 "Coding: clarify → signature + test → approach + complexity → code → edge cases",
                 "Design: requirements & budgets first, numbers, alternatives, monitoring",
                 "Think aloud; ask when stuck; summarise at the end of each answer",
                 "After each round: write down the questions you were asked"]:
        st.checkbox(item, key=f"dayof-{item}")

st.subheader("After the interview")
st.markdown("Add every question you couldn't answer well to your notes and look it up the same day — then go back to "
            "the 6-month plan (Today page). Mock results and card ratings already feed the plan.")
