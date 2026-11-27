"""Error log: every bug you hit and every question you fumbled becomes a review card."""
import pandas as pd
import streamlit as st

from mlprep import content, errorlog, progress
from mlprep.ui import data

d = data()
st.title("Error log")
st.markdown(
    "Your own mistakes are the highest-value study material. Log every bug you hit in an exercise, every question you "
    "fumbled in a mock or a real interview, and every concept gap you notice. Entries marked *in review* become cards in "
    "the spaced-repetition deck (prompt on the front, your lesson on the back)."
)
weeks = content.weeks()
with st.form("new-error", clear_on_submit=True):
    c1, c2 = st.columns([1, 1])
    kind = c1.selectbox("Type", errorlog.KINDS)
    wk = c2.selectbox("Related week", [None] + list(weeks), format_func=lambda k: "—" if k is None else f"W{k:02d} · {weeks[k].title}")
    prompt = st.text_input("What happened / the question (becomes the card front)",
                           placeholder="e.g. Forgot to divide the loss by accum_steps with gradient accumulation")
    lesson = st.text_area("The fix or the correct answer, in your own words (card back)", height=110)
    tags = st.text_input("Tags (comma-separated)", placeholder="pytorch, training")
    in_review = st.checkbox("Add to the review deck", True)
    if st.form_submit_button("Add entry", type="primary") and prompt.strip():
        errorlog.add(d, kind, prompt, lesson, wk, [t.strip() for t in tags.split(",") if t.strip()], in_review)
        progress.save(d)
        st.success("Logged.")

log = d.get("errors", [])
if not log:
    st.info("No entries yet. The grader page can also add fumbled questions here.")
    st.stop()
df = pd.DataFrame(log)
f = st.multiselect("Filter by type", errorlog.KINDS)
view = df[df.kind.isin(f)] if f else df
edited = st.data_editor(view[["id", "date", "kind", "prompt", "lesson", "week", "in_review"]], hide_index=True,
                        width="stretch", disabled=["id", "date"], key="errlog-editor")
if st.button("Save changes"):
    byid = {e["id"]: e for e in log}
    for row in edited.to_dict("records"):
        byid[row["id"]].update(kind=row["kind"], prompt=row["prompt"], lesson=row["lesson"], in_review=bool(row["in_review"]))
    progress.save(d)
    st.success("Saved.")
st.bar_chart(df.kind.value_counts(), height=200)
st.caption("Patterns matter: if most entries are 'Bug I made' in PyTorch shapes, add shape asserts to your habits.")
