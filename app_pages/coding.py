"""Coding interview: data-structures-and-algorithms problems by pattern, timed practice, mocks and re-solves."""
import datetime as dt
import random
import time

import pandas as pd
import streamlit as st

from mlprep import coding, content, progress
from mlprep.runner import run_pytest
from mlprep.stubs import stub_source
from mlprep.ui import data, go_coding, go_exercise, md

d = data()
probs = coding.problems()
today = dt.date.today()
LEVEL_ICON = {"easy": "🟢", "medium": "🟠", "hard": "🔴"}

st.title("Coding interview")
st.markdown(
    f"{len(probs)} classic interview problems grouped by pattern, each with tests and a reference solution, plus "
    "the ML-from-scratch questions interviewers ask most. Two a week are scheduled alongside the main plan. Solve "
    "with the **timer** on: a clean solve (tests pass, within the target time, no hints) moves the problem up the "
    f"re-solve ladder ({' → '.join(str(x) for x in coding.LADDER)} days). A slow, hinted or failed solve brings it "
    "back sooner."
)


def status_icon(key: str) -> str:
    if coding.mastered(d, key):
        return "🏆"
    if coding.solved(d, key):
        return "✅"
    if key in coding.state(d):
        return "❌"
    return "⬜"


def label(key: str) -> str:
    p = probs[key]
    return f"{status_icon(key)} {LEVEL_ICON[p.level]} {p.title}{' ★' if p.star else ''}  ·  {p.pattern}"


due = coding.due(d, today)
solved_n = sum(coding.solved(d, k) for k in probs)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Solved", f"{solved_n} / {len(probs)}")
c2.metric("Mastered", sum(coding.mastered(d, k) for k in probs), help="Solved cleanly at least twice on the ladder")
c3.metric("Re-solves due", len(due))
c4.metric("★ very-high frequency solved", f"{sum(coding.solved(d, k) for k, p in probs.items() if p.star)} / "
          f"{sum(p.star for p in probs.values())}")

with st.expander("How to work a coding interview problem (read once, then follow it every time)"):
    md(coding.guide().get("The method", ""))

tab_practice, tab_mock, tab_patterns, tab_ml, tab_progress = st.tabs(
    ["Practice", "Timed mock", "Patterns", "ML coding (most asked)", "Progress"])


# ---- Practice ----------------------------------------------------------------------
def run_and_record(key: str, code: str | None, timer_key: str = "coding_timer") -> None:
    p = probs[key]
    if code is not None:
        p.stub_path.write_text(code, encoding="utf-8")
    with st.spinner("Running the tests…"):
        ok, out = run_pytest(p.test_path)
    timer = st.session_state.get(timer_key)
    timed = timer and timer.get("key") == key
    minutes = (time.time() - timer["t0"]) / 60 if timed else None
    hints = timer.get("hints", 0) if timed else st.session_state.get(f"hints-{key}", 0)
    if ok or timed:  # untimed failures aren't attempts, just test runs
        e = coding.record(d, key, ok, minutes, hints, today)
        progress.save(d)
        if ok and timed:
            st.session_state.pop(timer_key, None)
        when = dt.date.fromisoformat(e["next"])
        nxt = f"Next re-solve: **{when:%a %d %b}**."
    else:
        nxt = ""
    if ok:
        t = f" in {minutes:.1f} min (target {p.minutes})" if minutes is not None else " (untimed)"
        st.success(f"All tests passed{t}. {nxt}")
    else:
        st.error(f"Some tests failed. {nxt}")
    st.code(out, language="text")


with tab_practice:
    f1, f2, f3, f4 = st.columns([2, 1.2, 1.2, 1])
    pat = f1.multiselect("Patterns", coding.PATTERNS, placeholder="All patterns", key="cp-pat")
    lev = f2.multiselect("Level", coding.LEVELS, placeholder="Any level", key="cp-lev")
    stat = f3.selectbox("Status", ["all", "unsolved", "due for re-solve", "solved", "mastered"], key="cp-stat")
    star = f4.toggle("★ only", help="Only very-high-frequency problems", key="cp-star")

    due_keys = {p.key for p in due}

    def keep(p) -> bool:
        if pat and p.pattern not in pat:
            return False
        if lev and p.level not in lev:
            return False
        if star and not p.star:
            return False
        s = {"all": True, "unsolved": not coding.solved(d, p.key), "due for re-solve": p.key in due_keys,
             "solved": coding.solved(d, p.key), "mastered": coding.mastered(d, p.key)}
        return s[stat]

    keys = [k for k, p in probs.items() if keep(p)]
    pre = st.session_state.pop("coding_key", None)
    if pre in probs and pre not in keys:
        keys.insert(0, pre)
    if not keys:
        st.info("No problems match these filters.")
    else:
        if pre in keys:
            st.session_state["cp-select"] = pre
        elif st.session_state.get("cp-select") not in keys:
            st.session_state["cp-select"] = keys[0]
        key = st.selectbox("Problem", keys, key="cp-select",  # labels must not change between runs
                           format_func=lambda k: f"{LEVEL_ICON[probs[k].level]} {probs[k].title}"
                           f"{' ★' if probs[k].star else ''}  ·  {probs[k].pattern}")
        p = probs[key]
        st.header(p.title)
        e = coding.state(d).get(key)
        status = ("🏆 mastered" if coding.mastered(d, key) else "✅ solved" if coding.solved(d, key)
                  else "❌ attempted" if e else "⬜ not attempted")
        best = coding.best_minutes(d, key)
        st.caption(f"{status}" + (f" · best {best:.0f} min" if best is not None else "")
                   + (f" · next re-solve {e['next']}" if e and e.get("next") else "") + " · "
                   f"{p.pattern} · {p.level} · target {p.minutes} min · asked {p.frequency}"
                   f"{' ★' if p.star else ''} · scheduled week {p.week}, day {p.day} · tags: {', '.join(p.tags)}")
        md(p.spec)

        timer = st.session_state.get("coding_timer")
        tcol1, tcol2, tcol3 = st.columns([1, 1, 2], vertical_alignment="center")
        if timer and timer["key"] == key:
            if tcol1.button("Stop timer", icon=":material/stop:", key="cp-stop"):
                st.session_state.pop("coding_timer")
                st.rerun()

            @st.fragment(run_every=1.0)
            def clock(t0: float, target: int):
                el = time.time() - t0
                over = el > target * 60
                st.markdown(f"⏱ **{int(el // 60):02d}:{int(el % 60):02d}** / {target}:00"
                            + (" — :red[over target]" if over else ""))

            with tcol3:
                clock(timer["t0"], p.minutes)
        else:
            if tcol1.button("Start timer", type="primary", icon=":material/timer:", key="cp-start"):
                st.session_state["coding_timer"] = {"key": key, "t0": time.time(), "hints": 0}
                st.rerun()
            if timer:
                tcol3.caption(f"A timer is running for *{probs[timer['key']].title}*.")
        if tcol2.button("Reset code to blank stub", key="cp-reset", help="Start a re-solve from scratch"):
            p.stub_path.write_text(stub_source(p.solution_path.read_text(encoding="utf-8")), encoding="utf-8")
            st.session_state.pop(f"cp-code-{key}", None)
            st.rerun()

        hkey = f"hints-{key}"
        shown = st.session_state.get(hkey, 0)
        for i in range(shown):
            st.info(f"**Hint {i + 1}.** {p.hints[i]}")
        if shown < len(p.hints) and st.button(f"Show hint {shown + 1} of {len(p.hints)}", key=f"cp-hint-{key}-{shown}"):
            st.session_state[hkey] = shown + 1
            if timer and timer["key"] == key:
                timer["hints"] = timer.get("hints", 0) + 1
            st.rerun()

        code = st.text_area("Your solution (saved to the exercise file)", p.stub_path.read_text(encoding="utf-8"),
                            height=380, key=f"cp-code-{key}")
        if st.button("Save & run tests", type="primary", icon=":material/play_arrow:", key=f"cp-run-{key}"):
            run_and_record(key, code)
        st.caption(f"Prefer your own editor? Edit `exercises/{key}.py`, then run "
                   f"`pytest tests/exercises/test_{key}.py`, or press the button here to log the attempt.")

        attempted = key in coding.state(d)
        with st.expander("Complexity & follow-ups" + ("" if attempted else " (after your first attempt)")):
            if attempted:
                st.markdown(f"**Target complexity:** {p.complexity}")
                st.markdown("**Follow-ups an interviewer may ask:**\n" + "\n".join(f"- {f}" for f in p.followups))
            else:
                st.caption("Make an attempt first: stating the complexity yourself is part of the exercise.")
        with st.expander("Tests"):
            st.code(p.test_path.read_text(encoding="utf-8"), language="python")
        with st.expander("Reference solution"):
            if st.toggle("I've made a real attempt — show it", key=f"cp-sol-{key}"):
                st.code(p.solution_path.read_text(encoding="utf-8"), language="python")


# ---- Timed mock --------------------------------------------------------------------------
with tab_mock:
    mock = st.session_state.get("coding_mock")
    if not mock:
        st.markdown("A realistic round: problems you haven't seen today, a hard time limit, no hints. Talk through the "
                    "method out loud as if an interviewer were listening.")
        m1, m2, m3 = st.columns(3)
        n = m1.radio("Problems", [1, 2], index=1, horizontal=True, key="cm-n")
        minutes = m2.select_slider("Minutes", [20, 30, 45, 60], value=45, key="cm-min")
        mix = m3.selectbox("Difficulty", ["easy + medium", "medium", "medium + hard", "any"], index=1, key="cm-mix")
        source = st.radio("Draw from", ["unsolved first", "due re-solves first", "any"], horizontal=True, key="cm-src")
        if st.button("Start mock", type="primary", icon=":material/rocket_launch:", key="cm-start"):
            levels = {"easy + medium": {"easy", "medium"}, "medium": {"medium"}, "medium + hard": {"medium", "hard"},
                      "any": set(coding.LEVELS)}[mix]
            pool = [k for k, p in probs.items() if p.level in levels and p.pattern != "ML from scratch"]
            rng = random.Random()
            rng.shuffle(pool)
            if source == "unsolved first":
                pool.sort(key=lambda k: coding.solved(d, k))
            elif source == "due re-solves first":
                pool.sort(key=lambda k: k not in due_keys)
            st.session_state["coding_mock"] = {"keys": pool[:n], "t0": time.time(), "minutes": minutes, "passed": {}}
            st.rerun()
    else:
        el = time.time() - mock["t0"]
        left = mock["minutes"] * 60 - el

        @st.fragment(run_every=1.0)
        def countdown(t0: float, total: int):
            rem = total * 60 - (time.time() - t0)
            if rem > 0:
                st.markdown(f"### ⏳ {int(rem // 60):02d}:{int(rem % 60):02d} left")
            else:
                st.markdown("### ⌛ Time's up — finish the test you're on, then wrap up.")

        countdown(mock["t0"], mock["minutes"])
        for i, key in enumerate(mock["keys"]):
            p = probs[key]
            with st.container(border=True):
                done = key in mock["passed"]
                st.subheader(f"Problem {i + 1}: {p.title}" + (" ✅" if done else ""))
                st.caption(f"{p.level} · target {p.minutes} min")
                md(p.spec)
                code = st.text_area("Solution", p.stub_path.read_text(encoding="utf-8"), height=300, key=f"cm-code-{key}")
                if st.button("Run tests", key=f"cm-run-{key}"):
                    p.stub_path.write_text(code, encoding="utf-8")
                    with st.spinner("Running…"):
                        ok, out = run_pytest(p.test_path)
                    if ok and not done:
                        mock["passed"][key] = (time.time() - mock["t0"]) / 60
                    (st.success if ok else st.error)("Passed." if ok else "Failing — keep going.")
                    st.code(out, language="text")
        st.markdown("**Self-review** (be honest; this is what interviewers score):")
        r1, r2, r3, r4 = st.columns(4)
        crit = {"clarified": r1.checkbox("Clarified & gave examples", key="cm-c1"),
                "brute": r2.checkbox("Stated brute force + target complexity", key="cm-c2"),
                "tested": r3.checkbox("Tested edge cases by hand", key="cm-c3"),
                "complexity": r4.checkbox("Gave final complexity", key="cm-c4")}
        if st.button("Finish mock", type="primary", key="cm-finish"):
            for key in mock["keys"]:
                passed = key in mock["passed"]
                coding.record(d, key, passed, mock["passed"].get(key), 0, today)
            d.setdefault("coding_mocks", []).append({
                "date": today.isoformat(), "keys": mock["keys"], "minutes": mock["minutes"],
                "passed": {k: round(v, 1) for k, v in mock["passed"].items()},
                "elapsed": round(el / 60, 1), **crit})
            progress.save(d)
            st.session_state.pop("coding_mock")
            st.success(f"Logged: {len(mock['passed'])}/{len(mock['keys'])} solved in {el / 60:.0f} min.")
            st.rerun()
        if st.button("Abandon (don't log)", key="cm-abandon"):
            st.session_state.pop("coding_mock")
            st.rerun()
    hist = d.get("coding_mocks", [])
    if hist:
        st.subheader("Past mocks")
        st.dataframe(pd.DataFrame([{
            "Date": h["date"], "Solved": f"{len(h['passed'])}/{len(h['keys'])}", "Minutes used": h["elapsed"],
            "Problems": ", ".join(probs[k].title.split(" — ")[0] for k in h["keys"] if k in probs),
            "Process": sum(h.get(c, False) for c in ("clarified", "brute", "tested", "complexity")),
        } for h in reversed(hist)]), hide_index=True, width="stretch")


# ---- Patterns ----------------------------------------------------------------------------
with tab_patterns:
    pick = st.selectbox("Pattern", coding.PATTERNS, key="pat-pick")
    md(coding.guide().get(pick, "_No notes yet._"))
    st.subheader("Problems in this pattern")
    for key, p in probs.items():
        if p.pattern != pick:
            continue
        a, b = st.columns([0.8, 0.2], vertical_alignment="center")
        a.markdown(f"{label(key)}  \n:gray[target {p.minutes} min · week {p.week}]")
        if b.button("Solve", key=f"pat-go-{key}", width="stretch"):
            go_coding(key)


# ---- ML coding -----------------------------------------------------------------------------
with tab_ml:
    st.markdown(
        "The ML implementation questions that come up most in ML-engineer loops, with what the interviewer is "
        "actually checking. Each links to an exercise with tests. Aim to do every **very high** one in the target "
        "time without notes before an interview."
    )
    exs = content.exercises()
    ex_status = d.get("exercise_status", {})
    roles = sorted({r for *_, r in coding.ML_MOST_ASKED})
    role = st.multiselect("Role focus", roles, placeholder="All roles", key="ml-role")
    for key, freq, check, mins, r in coding.ML_MOST_ASKED:
        if role and r not in role:
            continue
        if key in probs:
            title, ok = probs[key].title, coding.solved(d, key)
        else:
            title, ok = exs[key].title, ex_status.get(key, {}).get("ok", False)
        a, b = st.columns([0.82, 0.18], vertical_alignment="center")
        a.markdown(f"{'✅' if ok else '⬜'} **{title}** · {freq} · ~{mins} min · *{r}*  \n:gray[{check}]")
        if b.button("Open", key=f"ml-go-{key}", width="stretch"):
            (go_coding if key in probs else go_exercise)(key)


# ---- Progress -------------------------------------------------------------------------------
with tab_progress:
    if due:
        st.subheader(f"Due for re-solve ({len(due)})")
        for p in due:
            a, b = st.columns([0.8, 0.2], vertical_alignment="center")
            a.markdown(f"{label(p.key)}  \n:gray[due {coding.state(d)[p.key]['next']}]")
            if b.button("Re-solve", key=f"due-{p.key}", width="stretch"):
                go_coding(p.key)
    else:
        st.caption("Nothing due for re-solve today.")

    st.subheader("By pattern")
    rows = []
    for pat in coding.PATTERNS:
        ks = [k for k, p in probs.items() if p.pattern == pat]
        if not ks:
            continue
        best = [(coding.best_minutes(d, k), probs[k].minutes) for k in ks if coding.best_minutes(d, k) is not None]
        rows.append({"Pattern": pat, "Problems": len(ks), "Solved": sum(coding.solved(d, k) for k in ks),
                     "Mastered": sum(coding.mastered(d, k) for k in ks),
                     "Best time vs target": f"{sum(b for b, _ in best) / max(1, sum(t for _, t in best)):.0%}" if best else "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    log = [(k, a) for k, e in coding.state(d).items() if k in probs for a in e["attempts"]]
    if log:
        st.subheader("Attempt log")
        st.dataframe(pd.DataFrame([{
            "Date": a["date"], "Problem": probs[k].title, "Passed": "✅" if a["passed"] else "❌",
            "Minutes": a["minutes"] if a["minutes"] is not None else "untimed", "Target": probs[k].minutes,
            "Hints": a["hints"],
        } for k, a in sorted(log, key=lambda t: t[1]["date"], reverse=True)]), hide_index=True, width="stretch")
