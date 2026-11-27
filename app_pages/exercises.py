import datetime as dt
import os
import subprocess
import sys

import streamlit as st

from mlprep import content, progress
from mlprep.content import ROOT
from mlprep.ui import data, md

d = data()
exs = content.exercises()
weeks = content.weeks()

st.title("Coding exercises")
st.markdown(
    "Each exercise is a stub in `exercises/` with a spec in its docstring and a pytest file in "
    "`tests/exercises/`. Implement it in your editor (or in the box below), then **Run tests**. "
    "Reference solutions live in `solutions/` — open them only after a real attempt, and then re-solve the "
    "exercise from scratch a few days later."
)

status = d["exercise_status"]
with st.sidebar:
    st.subheader("Filter")
    wk_filter = st.multiselect("Weeks", list(weeks), format_func=lambda k: f"W{k:02d} {weeks[k].title[:28]}")
    tags = sorted({t for e in exs.values() for t in e.tags})
    tag_filter = st.multiselect("Tags", tags)
    st_filter = st.radio("Status", ["all", "not passed", "passed"], horizontal=True)

def keep(e):
    if wk_filter and e.week not in wk_filter:
        return False
    if tag_filter and not set(tag_filter) & set(e.tags):
        return False
    passed = status.get(e.key, {}).get("ok", False)
    return st_filter == "all" or (st_filter == "passed") == passed

visible = [e for e in exs.values() if keep(e)]
n_pass = sum(1 for e in exs.values() if status.get(e.key, {}).get("ok"))
st.progress(n_pass / max(1, len(exs)), text=f"{n_pass} of {len(exs)} exercises passing")
if not visible:
    st.warning("No exercises match the filter.")
    st.stop()

pre = st.session_state.pop("exercise_key", None)
keys = [e.key for e in visible]
if pre and pre not in keys:
    keys.insert(0, pre)
key = st.selectbox("Exercise", keys, index=keys.index(pre) if pre in keys else 0,
                   format_func=lambda k: f"{'✅' if status.get(k, {}).get('ok') else '⬜'} W{exs[k].week:02d}·D"
                   f"{exs[k].day}  {exs[k].title}  ({exs[k].level}, ~{exs[k].minutes} min)")
ex = exs[key]

st.header(ex.title)
st.caption(f"Week {ex.week}, day {ex.day} · {ex.level} · ~{ex.minutes} min · tags: {', '.join(ex.tags)}"
           + (" · ★ must-know" if ex.core else ""))
md(ex.spec)
st.markdown(f"Files: `exercises/{key}.py` (edit this) · `tests/exercises/test_{key}.py` · "
            f"`solutions/{key}.py`")

tab_edit, tab_tests, tab_sol = st.tabs(["Edit & run", "Tests", "Reference solution"])

def run_tests() -> tuple[bool, str]:
    env = dict(os.environ, MLPREP_TARGET="exercises")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider",
                        str(ex.test_path)], cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
    return r.returncode == 0, (r.stdout + r.stderr)[-6000:]

with tab_edit:
    code = st.text_area("Your code (saved to the exercise file)", ex.stub_path.read_text(encoding="utf-8"),
                        height=420, key=f"code-{key}")
    c1, c2, _ = st.columns([0.2, 0.2, 0.6])
    if c1.button("Save & run tests", type="primary", icon=":material/play_arrow:"):
        ex.stub_path.write_text(code, encoding="utf-8")
        with st.spinner("Running pytest…"):
            ok, out = run_tests()
        status[key] = {"ok": ok, "when": dt.datetime.now().isoformat(timespec="minutes")}
        progress.save(d)
        (st.success if ok else st.error)("All tests passed." if ok else "Some tests failed — see output.")
        st.code(out, language="text")
    if c2.button("Run tests only"):
        with st.spinner("Running pytest…"):
            ok, out = run_tests()
        status[key] = {"ok": ok, "when": dt.datetime.now().isoformat(timespec="minutes")}
        progress.save(d)
        (st.success if ok else st.error)("All tests passed." if ok else "Some tests failed — see output.")
        st.code(out, language="text")
    if key in status:
        st.caption(f"Last run {status[key]['when']}: {'passed' if status[key]['ok'] else 'failed'}")
    st.caption("Prefer your own editor? Run `pytest tests/exercises/test_" + key + ".py` from the repo root.")

with tab_tests:
    st.code(ex.test_path.read_text(encoding="utf-8"), language="python")

with tab_sol:
    if st.toggle("I've made a real attempt — show the reference solution", key=f"sol-{key}"):
        st.code(ex.solution_path.read_text(encoding="utf-8"), language="python")
