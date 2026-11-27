"""System design bank: 14 embedded/audio/edge designs to practise out loud, with progressive reveal."""
import re
import time

import streamlit as st

from mlprep import progress
from mlprep.content import ROOT
from mlprep.ui import data, go_theory, md

d = data()
src = (ROOT / "content" / "system_design.md").read_text(encoding="utf-8")
intro, *blocks = re.split(r"^## ", src, flags=re.M)
designs = []
for b in blocks:
    head, body = b.split("\n", 1)
    slug, title, tags = [x.strip() for x in head.split("|")]
    secs = dict(re.findall(r"^### (.+?)\n(.*?)(?=^### |\Z)", body, flags=re.S | re.M))
    designs.append({"slug": slug, "title": title, "tags": tags, "secs": {k: v.strip() for k, v in secs.items()}})

st.title("System design bank")
st.markdown(intro.strip() + "  \nFive more designs (wake word, hearing-aid NS, on-device assistant, DMS, fleet sensor ML) are "
            "worked through in week 26.")
if st.button("Open week 26 — the design framework"):
    go_theory(26, 1)
done = d.setdefault("designs_done", {})
pick = st.selectbox("Design", [x["slug"] for x in designs],
                    format_func=lambda s: f"{'✅' if s in done else '⬜'} {next(x for x in designs if x['slug'] == s)['title']}")
x = next(x for x in designs if x["slug"] == pick)
st.header(x["title"])
st.caption(x["tags"])
with st.container(border=True):
    st.markdown("**Prompt**")
    md(x["secs"].get("Prompt", ""))

c1, c2 = st.columns([1, 3], vertical_alignment="center")
if c1.button("Start 45-min timer", icon=":material/timer:"):
    st.session_state[f"t0-{pick}"] = time.time()
t0 = st.session_state.get(f"t0-{pick}")
if t0:
    @st.fragment(run_every=5)
    def show():
        left = 45 * 60 - int(time.time() - t0)
        st.markdown(f"⏱ {max(0, left) // 60}:{max(0, left) % 60:02d} left" + ("  :red[time]" if left <= 0 else ""))
    with c2:
        show()
notes = st.text_area("Your design notes (saved)", d.setdefault("design_notes", {}).get(pick, ""), height=200, key=f"n-{pick}")
if notes != d["design_notes"].get(pick, ""):
    d["design_notes"][pick] = notes
    progress.save(d)

st.caption("Reveal one section at a time, after you've covered it yourself.")
for sec in ["Clarify first", "Key numbers", "Reference design", "Trade-offs & pitfalls", "Follow-ups"]:
    if sec in x["secs"]:
        with st.expander(sec):
            md(x["secs"][sec])
if st.checkbox("I've done this design out loud", value=pick in done, key=f"done-{pick}"):
    if pick not in done:
        done[pick] = True
        progress.save(d)
elif pick in done:
    done.pop(pick)
    progress.save(d)
