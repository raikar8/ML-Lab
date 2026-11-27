"""Papers: the whole reading list in one place, with search, filters, read tracking and five-line notes."""
import datetime as dt

import streamlit as st

from mlprep import content, papers, plan, progress
from mlprep.ui import data, go_theory

d = data()
allp = papers.all_papers()
weeks = content.weeks()
electives = content.optional_weeks()

st.title("Papers")
st.markdown(
    "Every primary source in the plan — papers, technical reports, a few author blogs and docs — tagged by week and "
    "topic. Tick a paper when you've read it and write the five-line note (what changed, why it works, evidence, "
    "cost, one question): saying those five things out loud is exactly what an interviewer asks when you mention a "
    "paper. Links go to arXiv or the publisher; nothing is mirrored here."
)

# ---- headline numbers -------------------------------------------------------
start = progress.start_date(d)
cur = plan.week_day_for(start, dt.date.today())
cur_sec = f"W{cur[0]}" if cur else None
read_n = sum(papers.is_read(d, p.pid) for p in allp)
notes_n = sum(bool(papers.note(d, p.pid)) for p in allp)
c1, c2, c3 = st.columns(3)
c1.metric("Read", f"{read_n} / {len(allp)}")
c2.metric("Five-line notes", notes_n)
if cur_sec:
    this_week = [p for p in allp if p.section == cur_sec]
    c3.metric(f"This week (W{cur[0]})", f"{sum(papers.is_read(d, p.pid) for p in this_week)} / {len(this_week)}")


# ---- filters ------------------------------------------------------------------
def _section_label(s: str) -> str:
    if s.startswith("W"):
        return f"W{int(s[1:]):02d} · {weeks[int(s[1:])].title}" if int(s[1:]) in weeks else s
    return f"{s} · {electives[s].title}" if s in electives else s


sections = list(dict.fromkeys(p.section for p in allp))
preset = st.session_state.pop("papers_section", None)
if preset in sections:
    st.session_state["papers_sections"] = [preset]

f1, f2 = st.columns([2, 1])
q = f1.text_input("Search title, authors or summary", placeholder="e.g. MLA, GRPO, quantization, Moonshot")
status = f2.segmented_control("Show", ["All", "Unread", "Read", "With notes"], default="All") or "All"
f3, f4, f5 = st.columns([2, 2, 1])
sel_sections = f3.multiselect("Weeks", sections, format_func=_section_label, key="papers_sections",
                              placeholder="All weeks")
sel_topics = f4.multiselect("Topics", papers.TOPICS, placeholder="All topics")
sel_source = f5.multiselect("Source", sorted({p.source for p in allp}), placeholder="Any")


def _keep(p: papers.Paper) -> bool:
    if sel_sections and p.section not in sel_sections:
        return False
    if sel_topics and not set(sel_topics) & set(p.topics):
        return False
    if sel_source and p.source not in sel_source:
        return False
    if q and q.lower() not in f"{p.title} {p.meta} {p.why}".lower():
        return False
    r, n = papers.is_read(d, p.pid), bool(papers.note(d, p.pid))
    return {"All": True, "Unread": not r, "Read": r, "With notes": n}[status]


shown = [p for p in allp if _keep(p)]
st.caption(f"{len(shown)} of {len(allp)} shown")


# ---- list -----------------------------------------------------------------------
def _on_read(pid: str) -> None:
    papers.set_read(d, pid, st.session_state[f"read-{pid}"])
    progress.save(d)


def paper_card(p: papers.Paper) -> None:
    with st.container(border=True):
        c1, c2 = st.columns([0.06, 0.94], vertical_alignment="top")
        c1.checkbox("read", value=papers.is_read(d, p.pid), key=f"read-{p.pid}", on_change=_on_read,
                    args=(p.pid,), label_visibility="collapsed", help="Mark as read")
        when = papers.state(d).get(p.pid, {}).get("read")
        c2.markdown(f"**[{p.title}]({p.url})**  \n{p.meta} · :gray[{p.source}]" + (f" · :green[read {when}]" if when else ""))
        if p.why:
            c2.markdown(p.why)
        c2.caption(" · ".join(p.topics))
        existing = papers.note(d, p.pid)
        with c2.expander("Five-line note" + (" ✓" if existing else "")):
            with st.form(f"note-{p.pid}", border=False):
                vals = {k: st.text_input(label, value=existing.get(k, ""), placeholder=hint, key=f"n-{p.pid}-{k}")
                        for k, label, hint in papers.NOTE_FIELDS}
                if st.form_submit_button("Save note"):
                    papers.set_note(d, p.pid, vals)
                    if any(v.strip() for v in vals.values()) and not papers.is_read(d, p.pid):
                        papers.set_read(d, p.pid, True)
                    progress.save(d)
                    st.rerun()


for sec in dict.fromkeys(p.section for p in shown):
    group = [p for p in shown if p.section == sec]
    h1, h2 = st.columns([0.8, 0.2], vertical_alignment="center")
    h1.subheader(_section_label(sec))
    if sec.startswith("W") and h2.button("Open week", key=f"open-{sec}", width="stretch"):
        go_theory(int(sec[1:]))
    for p in group:
        paper_card(p)

if not shown:
    st.info("Nothing matches these filters.")

st.divider()
st.download_button("Download all notes (Markdown)", papers.notes_markdown(d), file_name="paper_notes.md",
                   mime="text/markdown", disabled=notes_n == 0, icon=":material/download:")
